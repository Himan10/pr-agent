import re
import os
import pprint
import datetime
import json
import markdown
from colorama import Fore, Style
from typing import List, Dict, Any
from functools import partial
from jinja2 import Environment, StrictUndefined
from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.ai_handlers.openai_ai_handler import OpenAIHandler
from pr_agent.git_providers import get_git_provider_with_context
from pr_agent.git_providers.git_provider import (IncrementalPR, get_main_pr_language)
from pr_agent.log import get_logger
from pr_agent.config_loader import get_settings
from pr_agent.tools.scan_pr_contents import SecurityScanner, SCA_FILES_FORMAT
from pr_agent.algo.utils import ModelType
from pr_agent.algo.pr_processing import retry_with_fallback_models
from pr_agent.algo.git_patch_processing import decouple_and_convert_to_hunks_with_lines_numbers

class PRSecurityReview(SecurityScanner):
    """
    This class is responsible to run security checks on the PR/MR. 
    Security checks:
    1. Secrets - run noseyparker
    2. SAST - run semgrep (auto/default-rules)
    3. SBOM - run syft or trivy (here, we are using syft as of now)
    """

    def __init__(self, pr_url: str, is_answer: bool = False, is_auto: bool = False, args: list = None, ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler):
        """Initialize the PRSecurityReview class for running security checks on given PR"""

        self.git_provider = get_git_provider_with_context(pr_url) # returns an instance of the following: Gitlab/Github/bitbucket/localGitProvider
        self.args = args
        # scan_type would contain the first argument of "args"
        self.scan_type = self.args[0].lower().strip() # possible values: "secrets" | "sast" | "sca"
        self.incremental = self.parse_incremental(args)  # -i command | need to dig more deeper
        if self.incremental and self.incremental.is_incremental:
            self.git_provider.get_incremental_commits(self.incremental)

        self.main_language = get_main_pr_language(
            self.git_provider.get_languages(), self.git_provider.get_files()
        )
        self.pr_url = pr_url
        self.is_answer = is_answer
        self.is_auto = is_auto

        if self.is_answer and not self.git_provider.is_supported("get_issue_comments"):
            raise Exception(f"Answer mode is not supported for {get_settings().config.git_provider} for now")
        self.ai_handler = ai_handler()
        self.ai_handler.main_pr_language = self.main_language
        self.patches_diff = None
        self.prediction = None
        self.logger = get_logger()
        #answer_str, question_str = self._get_user_answers()
        self.pr_description, self.pr_description_files = (
            self.git_provider.get_pr_description(split_changes_walkthrough=True))
        if (self.pr_description_files and get_settings().get("config.is_auto_command", False) and
                get_settings().get("config.enable_ai_metadata", False)):
            add_ai_metadata_to_diff_files(self.git_provider, self.pr_description_files)
            self.logger.debug(f"AI metadata added to the this command")
        else:
            get_settings().set("config.enable_ai_metadata", False)
            self.logger.debug(f"AI metadata is disabled for this command")

        # instantiate the parent class - SecurityScanner
        super().__init__(self.get_new_hunks_with_line_numbers())
        # findings definition
        self.finding = {
            "data": None
        }
        self.findings_summary = None
        self.scan_results = None

        # declare a dict of key:values to supply to other methods
        self.vars = {
            "title": self.git_provider.pr.title,
            "branch": self.git_provider.get_pr_branch(),
            "description": self.pr_description,
            "language": self.main_language,
            "diff": "",  # empty diff for initial calculation
            "num_pr_files": self.git_provider.get_num_of_files(),
            "num_max_findings": get_settings().pr_reviewer.num_max_findings,
            "require_score": get_settings().pr_reviewer.require_score_review,
            "require_tests": get_settings().pr_reviewer.require_tests_review,
            "require_estimate_effort_to_review": get_settings().pr_reviewer.require_estimate_effort_to_review,
            'require_can_be_split_review': get_settings().pr_reviewer.require_can_be_split_review,
            'require_security_review': get_settings().pr_reviewer.require_security_review,
            'require_todo_scan': get_settings().pr_reviewer.get("require_todo_scan", False),
            "extra_instructions": get_settings().pr_reviewer.extra_instructions,
            "commit_messages_str": self.git_provider.get_commit_messages(),
            "custom_labels": "",
            "enable_custom_labels": get_settings().config.enable_custom_labels,
            "is_ai_metadata":  get_settings().get("config.enable_ai_metadata", False),
            "related_tickets": get_settings().get('related_tickets', []),
            'duplicate_prompt_examples': get_settings().config.get('duplicate_prompt_examples', False),
            "date": datetime.datetime.now().strftime('%Y-%m-%d'),
        }

    def __str__(self) -> None:
        return f"PR Title: {self.vars["title"]}\nBranch: {self.vars["branch"]}\nDescription: {self.vars["description"]}\nLanguages: {self.vars["language"]}"

    def _get_added_lines_content(self) -> dict:
        """
        fetch only the added lines (+ lines) from all diff files in the PR.
        """
        added_content = {}
        
        try:
            diff_files = self.git_provider.get_diff_files()
            
            for file_info in diff_files:
                filename = file_info.filename
                patch = file_info.patch
                
                if not patch:
                    continue
                    
                added_lines = []
                patch_lines = patch.splitlines()
                
                for line in patch_lines:
                    # Only include lines that start with '+' but not '++'
                    # '++' indicates file headers in unified diff format
                    if line.startswith('+') and not line.startswith('+++'):
                        # Remove the '+' prefix to get the actual content
                        added_lines.append(line[1:])
                
                if added_lines:
                    added_content[filename] = added_lines
                    
        except Exception as e:
            self.logger.error(f"Error extracting added lines: {e}")
            
        return added_content
    
    def get_added_lines_as_text(self) -> dict:
        """
        fetch only the added lines from diff files and return as concatenated text per file.
        """
        added_content = self._get_added_lines_content()
        
        added_text = {}
        for filename, lines in added_content.items():
            added_text[filename] = '\n'.join(lines) + '\n' if lines else ''
            
        return added_text

    def get_new_hunks_with_line_numbers(self) -> dict:
        """
        fetch only the new hunks (added content) with line numbers for security analysis.
        """
        
        new_hunks = {}
        diff_files = self.git_provider.get_diff_files()
        
        for file_info in diff_files:
            if os.path.basename(file_info.filename) in SCA_FILES_FORMAT or self.scan_type.lower() == "sast":
                self.logger.info(f"SKIP processing hunk for file: {file_info.filename}")
                new_hunks[file_info.filename] = file_info.head_file
                continue
            if not file_info.patch:
                continue
                
            # Get the formatted hunks
            self.logger.info(f"processing hunk for file: {file_info.filename}")
            formatted_hunks = decouple_and_convert_to_hunks_with_lines_numbers(file_info.patch, file_info)
            
            # Extract only new hunk sections using regex
            new_hunk_pattern = r'__new hunk__\n(.*?)(?=__old hunk__|$)'
            new_hunk_matches = re.findall(new_hunk_pattern, formatted_hunks, re.DOTALL)
            
            if new_hunk_matches:
                new_hunks[file_info.filename] = '\n'.join(new_hunk_matches).strip()
                
        return new_hunks
    
    async def run(self) -> dict:
        """
        main function to run, this will call get_security_review
        """

        try:
            if not self.git_provider.get_files():
                self.logger.info(f"PR has no files: {self.pr_url}, skipping review")
                return None
            
            self.logger.info(f"Running {self.scan_type} scan for PR: {self.pr_url}")
            
            if self.scan_type == "secrets":
                results = await self.secrets()
                
            elif self.scan_type == "sast":
                results = await self.sast()
                
            elif self.scan_type == "sca":
                results = await self.sca()
                
            else:
                raise ValueError(f"Unsupported scan type: '{self.scan_type}'. Supported types are: 'secrets', 'sast', 'sca'")
            
            # Add PR metadata to results
            results["pr_metadata"] = {
                "pr_url": self.pr_url,
                "title": self.vars["title"],
                "branch": self.vars["branch"],
                "language": self.vars["language"],
                "scan_type": self.scan_type,
                "scan_timestamp": datetime.datetime.now().isoformat()
            }
            
            self.logger.info(f"{self.scan_type.upper()} scan completed for PR: {self.pr_url}")
            
            # Send the result to openai handler to retrieve details about the findings
            # get_prediction = True, means it will scan the findings_summary
            # if get_prediction = False, it'll analyze vulnerable files found by the scanners

            choice = input("Open interactive vulnerability viewer? [y/n]: ").lower()
            await self.get_security_review(results, get_prediction=False if choice.strip() == 'y' else True)
            if choice.strip() == 'y':
                # interactive viewer allows viewing and analyzing files repeatedly
                await self.interactive_vulnerability_viewer()
            # write the output file to markdown
            # write output to a file based on user's choice
            user_choice = input("Would you like to save the raw output in a file? [y/n]: ")
            if user_choice.lower() == 'y':
                self._write_scan_results_to_file(results)
            #report_md = convert_to_markdown(self.prediction)
            return
            
        except ValueError as e:
            self.logger.error(str(e))
            raise e
        except Exception as e:
            self.logger.error(f"{self.scan_type.upper()} scan failed for PR {self.pr_url}: {e}")
            return {
                "error": str(e),
                "pr_metadata": {
                    "pr_url": self.pr_url,
                    "scan_type": self.scan_type,
                    "scan_timestamp": datetime.datetime.now().isoformat()
                }
            }

    def parse_incremental(self, args: List[str]):
        is_incremental = False
        if args and len(args) >= 1:
            arg = args[0]
            if arg == "-i":
                is_incremental = True
        incremental = IncrementalPR(is_incremental)
        return incremental

    async def get_security_review(self, scan_results: Dict[str, Any], get_prediction=False) -> Dict[str, Any]:
        """
        use openai handler to analyze security findings and provide fix recommendations and impact assessment.
        """
        try:
            # Extract findings from scan results
            self.findings_summary = self._extract_findings_summary(scan_results, display_output=True)
            #print(self.findings_summary)
            self.scan_results = scan_results
            
            if not self.findings_summary["has_findings"]:
                return {
                    "analysis": "No security findings detected in the scan results.",
                    "recommendations": [],
                    "pr_metadata": scan_results.get("pr_metadata", {})
                }
            
            # Use retry mechanism with fallback models like pr_reviewer.py
            # in case of configuration for the existing models, change the model in configuration.toml
            if get_prediction:
                await retry_with_fallback_models(self._prepare_prediction, model_type=ModelType.REGULAR)
                
                if not self.prediction:
                    return {
                        "error": "Failed to generate AI analysis with any available model",
                        "pr_metadata": scan_results.get("pr_metadata", {}),
                        "analysis_timestamp": datetime.datetime.now().isoformat()
                    }
                
                # This prediction will later be saved in the md file
                #self.logger.info(f"analysis: {self.prediction}")
                return self.prediction
            else:
                self.logger.warning("prediction is currently set to false")
                return None
            
        except Exception as e:
            self.logger.error(f"AI security analysis failed for PR {self.pr_url}: {e}")
            return {
                "error": str(e),
                "pr_metadata": scan_results.get("pr_metadata", {}),
                "analysis_timestamp": datetime.datetime.now().isoformat()
            }

    async def _prepare_prediction(self, model: str) -> None:
        """
        Prepare the AI prediction for security analysis using the specified model.
        """
        try:
            # Create template variables for prompt rendering
            if self.finding["data"]:
                scan_data = self.finding.get("data")
            else:
                scan_data = self.scan_results
            variables = {
                "title": self.vars.get('title', 'N/A'),
                "branch": self.vars.get('branch', 'N/A'),
                "language": self.vars.get('language', 'N/A'),
                "findings_summary": json.dumps(scan_data, indent=2) #self.findings_summary was used initially
            }
            
            # Load prompts from TOML file using Jinja2 templating
            environment = Environment(undefined=StrictUndefined)
            system_prompt = environment.from_string(get_settings().pr_security_review_prompt.system).render(variables)
            user_prompt = environment.from_string(get_settings().pr_security_review_prompt.user).render(variables)

            # Get AI analysis
            response, finish_reason = await self.ai_handler.chat_completion(
                model=model,
                temperature=0.2,  # Lower temperature for more consistent analysis
                system=system_prompt,
                user=user_prompt
            )
            
            # Parse AI response
            try:
                ai_analysis = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                ai_analysis = {
                    "raw_response": response,
                    "overall_risk_level": "unknown",
                    "summary": "AI analysis completed but response format was not JSON"
                }
            
            ai_analysis["pr_metadata"] = self.scan_results.get("pr_metadata", {})
            ai_analysis["analysis_timestamp"] = datetime.datetime.now().isoformat()
            
            self.prediction = ai_analysis
            self.logger.info(f"AI security analysis completed for PR: {self.pr_url}")
            report_md = convert_to_markdown(self.prediction)
            
        except Exception as e:
            self.logger.error(f"AI prediction preparation failed for model {model}: {e}")
            self.prediction = None
    
    def _deduplicate_findings(self, findings: List[Dict], dedup_keys: List[str]) -> List[Dict]:
        """
        Remove duplicate findings based on specified keys.
        Supports nested keys using dot notation
        For example: key = "matches.id" -> {"matches": {"id": 1}}
        """       
        def get_nested_value(obj: dict, key_path: str) -> str:
            """get value from nested dictionary using dot notation."""
            try:
                keys = key_path.split('.')
                value = obj
                for key in keys:
                    value = value.get(key, {})
                return str(value) if value else ""
            except (AttributeError, TypeError):
                return ""

        if not findings:
            return findings
            
        seen = set() # (Sample Output: seen = set((tupl1), (tupl2)))
        deduplicated = []
        
        for finding in findings:
            dedup_tuple = tuple(
                get_nested_value(finding, key) for key in dedup_keys
            )
            
            if dedup_tuple not in seen:
                seen.add(dedup_tuple)
                deduplicated.append(finding)
        
        if len(deduplicated) < len(findings):
            self.logger.info(f"Deduplicated {len(findings) - len(deduplicated)} duplicate findings")
            
        return deduplicated


    def _get_head_file_content(self, filename: str) -> str:
        """
        Retrieve the current head file content for a PR file path.
        Falls back to self.content_dict if head content is unavailable.
        """
        try:
            diff_files = self.git_provider.get_diff_files()
            for fo in diff_files:
                if fo.filename == filename:
                    # Prefer head_file when available
                    if getattr(fo, 'head_file', None):
                        return fo.head_file
                    break
        except Exception as e:
            self.logger.warning(f"Unable to fetch head content for {filename}: {e}")
        return self.content_dict.get(filename, "")

    async def _analyze_single_file(self, filename: str) -> Dict[str, Any]:
        """
        Send findings for a specific file to AI for analysis.
        """
        try:
            file_findings = self._get_file_findings(filename)
            if not file_findings["findings"]:
                return {"analysis": f"No findings for {filename}"}
            
            # Create simple prompt with findings
            findings_text = json.dumps(file_findings["findings"], indent=2)
            # add this findings_text to the self.finding
            if findings_text:
                self.finding["data"] = findings_text

            # Make call to openai handler
            await retry_with_fallback_models(self._prepare_prediction, model_type=ModelType.REGULAR)
            
            #response, _ = await self.ai_handler.chat_completion(
            #    model="gpt-4",
            #    temperature=0.2,
            #    system="You are a security analyst. Analyze the provided findings and give actionable recommendations.",
            #    user=prompt
            #)

            if not self.prediction:
                return {
                    "error": "Failed to generate AI analysis with any available model",
                    "pr_metadata": scan_results.get("pr_metadata", {}),
                    "analysis_timestamp": datetime.datetime.now().isoformat()
                }
        
            return self.prediction
            #return {"analysis": response, "filename": filename}
        except Exception as e:
            return {"error": str(e)}

    def _get_file_findings(self, filename: str) -> Dict[str, Any]:
        """
        Get findings for a specific file.
        """
        findings = {"filename": filename, "findings": []}
        
        try:
            if self.scan_type == "secrets":
                for finding in self.scan_results.get("secrets", []):
                    if "matches" in finding:
                        for match in finding["matches"]:
                            path_list = match.get("provenance", {})
                            if path_list:
                                for each_path in path_list:
                                    if each_path.get("path") == filename:
                                        findings["findings"].append({
                                            "type": "secret",
                                            "rule": match.get("rule_name"),
                                            "line": match.get("location", {}).get("offset_span", {}).get("start"),
                                            "snippet": match.get("snippet", {}).get("matching"),
                                            "filename": filename
                                })
            
            elif self.scan_type == "sast":
                for finding in self.scan_results.get("results", []):
                    if finding.get("path") == filename:
                        findings["findings"].append({
                            "type": "sast",
                            "rule_id": finding.get("check_id"),
                            "message": finding.get("message"),
                            "line": finding.get("extra", {}).get("lines"), #finding.get("start", {}).get("line"),
                            "message": finding.get("extra", {}).get("message"),
                            "severity": finding.get("extra", {}).get("severity"),
                            "filename": filename,
                            "line_number": finding.get("start", {}).get("line")
                        })
            
            elif self.scan_type == "sca" and os.path.basename(filename) in SCA_FILES_FORMAT:
                is_content_exist = False
                for vuln in self.scan_results.get("vulnerabilities", []):
                    # check for filename
                    is_file_present = filter(lambda x: os.path.basename(x.get("path")) == os.path.basename(filename), vuln.get("artifact", {}).get("locations"))
                    is_content_exist = list(is_file_present) or False
                    if is_content_exist:
                        findings["findings"].append({
                            "type": "sca",
                            "vuln_id": vuln.get("vulnerability", {}).get("id"),
                            "package": vuln.get("artifact", {}).get("name"),
                            "version": vuln.get("artifact", {}).get("version"),
                            "severity": vuln.get("vulnerability", {}).get("severity"),
                            "description": vuln.get("vulnerability", {}).get("description"),
                            "filename": filename
                        })
        
        except Exception as e:
            findings["error"] = str(e)
        
        #print(findings)
        return findings

    def _extract_findings_summary(self, scan_results: Dict, display_output: bool = False) -> Dict:
        """
        Extracts and summarizes scan results with counts and vulnerable filenames.
        """
        summary = {
            "has_findings": False,
            "secrets": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0
            },
            "sast": {
                "error": 0,
                "warning": 0,
                "note": 0,
                "info": 0,
                "low": 0
            },
            "sca": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "component_count": 0
            },
            "total_issues": 0,
            "vulnerable_files": []
        }
        summary["secrets"]["files"] = []
        summary["sast"]["files"] = []
        summary["sca"]["files"] = []
        
        # secrets severity breakdown
        if "secrets" == scan_results.get("pr_metadata", {}).get("scan_type"):
            if "secrets" in scan_results and not scan_results.get("error"):
                secrets_data = scan_results["secrets"]
                if isinstance(secrets_data, list):
                    files = set()
                    for finding in secrets_data:
                        if isinstance(finding, dict) and "matches" in finding:
                            for m in finding["matches"]:
                                path_list = m.get("provenance", {})
                                summary["secrets"]["high"] += 1
                                if path_list:
                                    for each_path in path_list:
                                        files.add(each_path.get("path"))
                    
                    summary["has_findings"] = True
                    summary["secrets"]["files"] = sorted(files)
        
        # SAST severity breakdown
        if "sast" == scan_results.get("pr_metadata", {}).get("scan_type"):
            if "results" in scan_results and not scan_results.get("sast", {}).get("error"):
                sast_data = scan_results["results"]
                if isinstance(sast_data, list):
                    # Process deduplicated SAST findings by severity
                    deduped_sast = self._deduplicate_findings(
                        sast_data, 
                        ["check_id", "path", "start.line", "start.column"]
                    )

                    files = set()
                    for finding in deduped_sast:
                        severity = finding.get("extra", {}).get("severity", "warning").lower()
                        if severity in summary["sast"]:
                            summary["sast"][severity] += 1
                        else:
                            summary["sast"]["warning"] += 1
                        
                        path = finding.get("path")
                        if path:
                            files.add(path)
                    
                    if len(deduped_sast) > 0:
                        summary["has_findings"] = True
                        summary["sast"]["files"] = sorted(files)
        
        # SCA severity breakdown
        if "sca" == scan_results.get("pr_metadata", {}).get("scan_type"):
            if "vulnerabilities" in scan_results and not scan_results.get("error"):
                vulnerabilities = scan_results["vulnerabilities"]
                sbom_components = scan_results.get("sbom", {}).get("components", [])
                
                # Set component count
                summary["sca"]["component_count"] = len(sbom_components)
                
                # Process vulnerabilities by severity
                if isinstance(vulnerabilities, list) and vulnerabilities:
                    # Deduplicate vulnerabilities first
                    deduped_vulns = self._deduplicate_findings(
                        vulnerabilities,
                        ["vulnerability.id", "artifact.name", "artifact.version"]
                    )
                    
                    # Count by severity
                    for vuln in deduped_vulns:
                        severity = vuln.get("vulnerability", {}).get("severity", "").lower()
                        if severity in summary["sca"]:
                            summary["sca"][severity] += 1
                        else:
                            # Default to medium if severity not recognized
                            summary["sca"]["medium"] += 1
                    
                    if len(deduped_vulns) > 0:
                        summary["has_findings"] = True
                        # For SCA, just list the dependency files
                        sca_files = [path for path in self.content_dict.keys() if os.path.basename(path) in SCA_FILES_FORMAT]
                        summary["sca"]["files"] = sorted(set(sca_files))
        
        summary["total_issues"] = (
            sum(v for k, v in summary["secrets"].items() if k != "files") + 
            sum(v for k, v in summary["sast"].items() if k != "files") + 
            sum(v for k, v in summary["sca"].items() if k not in ("component_count", "files"))
        )

        all_files = set()
        all_files.update(summary["secrets"]["files"])
        all_files.update(summary["sast"]["files"])
        all_files.update(summary["sca"]["files"])
        summary["vulnerable_files"] = sorted(all_files)
        
        # Display stylized output if requested
        if display_output:
            self._display_stylized_summary(summary, scan_results)
        
        return summary

    def _display_stylized_summary(self, summary: Dict, scan_results: Dict) -> None:
        """
        Display stylized summary output with colorama colors
        """
        from colorama import Fore, Back, Style, init
        init(autoreset=True)  # Auto-reset colors after each print
        
        # Color mapping
        severity_colors = {
            'critical': Fore.MAGENTA + Style.BRIGHT,    # Dark Red/Magenta
            'high': Fore.RED + Style.BRIGHT,            # Light Red
            'medium': Fore.YELLOW + Style.BRIGHT,       # Yellow
            'low': Fore.GREEN + Style.BRIGHT,           # Green
            'info': Fore.BLUE + Style.BRIGHT,           # Blue
            'error': Fore.RED + Style.BRIGHT,           # Red for errors
            'warning': Fore.YELLOW + Style.BRIGHT,      # Yellow for warnings
            'note': Fore.CYAN + Style.BRIGHT            # Cyan for notes
        }
        
        scan_type = scan_results.get("pr_metadata", {}).get("scan_type", "unknown")
        pr_title = scan_results.get("pr_metadata", {}).get("title", "N/A")
        
        print(f"\n{Style.BRIGHT}SECURITY SCAN SUMMARY{Style.RESET_ALL}")
        print("=" * 50)
        print(f"{Fore.CYAN}PR Title:{Style.RESET_ALL} {pr_title}")
        print(f"{Fore.CYAN}Scan Type:{Style.RESET_ALL} {scan_type.upper()}")
        print(f"{Fore.CYAN}Total Issues:{Style.RESET_ALL} {summary['total_issues']}")
        print(f"{Fore.CYAN}Vulnerable Files:{Style.RESET_ALL} {len(summary['vulnerable_files'])}")
        
        if not summary["has_findings"]:
            print(f"\n{Fore.GREEN + Style.BRIGHT}No security issues found!{Style.RESET_ALL}")
            return
        
        print(f"\n{Style.BRIGHT}FINDINGS BREAKDOWN{Style.RESET_ALL}")
        print("-" * 30)
        
        # Display findings by scan type
        if scan_type == "secrets":
            self._display_scan_type_summary("SECRETS", summary["secrets"], severity_colors)
        elif scan_type == "sast":
            self._display_scan_type_summary("SAST", summary["sast"], severity_colors)
        elif scan_type == "sca":
            self._display_scan_type_summary("SCA", summary["sca"], severity_colors, show_components=True)
        
        # Display vulnerable files
        if summary["vulnerable_files"]:
            print(f"\n{Style.BRIGHT}VULNERABLE FILES{Style.RESET_ALL}")
            print("-" * 20)
            for i, file_path in enumerate(summary["vulnerable_files"], 1):
                filename = os.path.basename(file_path)
                print(f"  {Fore.WHITE}{i:2d}.{Style.RESET_ALL} {Fore.YELLOW}{filename}{Style.RESET_ALL}")
        
        print(f"\n{Style.BRIGHT}NEXT STEPS{Style.RESET_ALL}")
        print("-" * 15)
        if summary["total_issues"] > 0:
            # Determine priority based on severity counts
            critical_total = (summary["secrets"]["critical"] + 
                            summary["sca"]["critical"])
            high_total = (summary["secrets"]["high"] + 
                         summary["sast"]["error"] + 
                         summary["sca"]["high"])
            
            if critical_total > 0:
                print(f"  {severity_colors['critical']}CRITICAL: Address {critical_total} critical issues immediately{Style.RESET_ALL}")
            if high_total > 0:
                print(f"  {severity_colors['high']}HIGH: Review {high_total} high-severity findings{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}Run interactive viewer for detailed analysis{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}Generate AI analysis for remediation guidance{Style.RESET_ALL}")
        
        print("=" * 50)

    def _display_scan_type_summary(self, title: str, findings: Dict, colors: Dict, show_components: bool = False) -> None:
        """
        Display summary for a specific scan type with colored output
        """
        
        print(f"\n{Style.BRIGHT}{title}{Style.RESET_ALL}")
        
        total_findings = 0
        for severity, count in findings.items():
            if severity == "files" or (show_components and severity == "component_count"):
                continue
            if count > 0:
                color = colors.get(severity, Fore.WHITE)
                print(f"  {color}{severity.upper()}: {count}{Style.RESET_ALL}")
                total_findings += count
        
        if show_components and "component_count" in findings:
            print(f"  {Fore.CYAN}Components Analyzed: {findings['component_count']}{Style.RESET_ALL}")
        
        if total_findings == 0:
            print(f"  {Fore.GREEN}No issues found{Style.RESET_ALL}")

    async def interactive_vulnerability_viewer(self) -> None:
        """
        Commands:
          - list: show indexed vulnerable files by scan type
          - view <file_index>: show vulnerability findings for the file
          - analyze <file_index>: re-scan and AI-analyze only that file
          - help: show commands
          - q/quit: exit
        """
        try:
            def print_list():
                self.logger.info("\nVulnerable files:")
                for idx, (st, f) in enumerate(entries, start=1):
                    self.logger.info(f"  [{idx}] ({st}) {os.path.basename(f)}")
                self.logger.info("")

            def show_findings(filename):
                findings = self._get_file_findings(filename)
                self.logger.info(f"\nFINDINGS FOR {os.path.basename(filename)}")
                
                if "error" in findings:
                    self.logger.error(f"Error: {findings['error']}")
                    return
                
                if not findings["findings"]:
                    self.logger.info("No findings for this file.")
                    return
                
                for i, f in enumerate(findings["findings"], 1):
                    formatted_items = []
                    for key, value in f.items():
                        formatted_items.append(f"{Fore.CYAN}{key}{Style.RESET_ALL}: {Fore.YELLOW}{value}{Style.RESET_ALL}")
                    #self.logger.info(f"[{i}] " + " | ".join(formatted_items))
                    print(" \n ".join(formatted_items))

            #print(self.findings_summary)
            if not getattr(self, "findings_summary", None):
                self.logger.info("No findings summary available. Run a scan first.")
                return

            # Prepare indexed list
            #print(self.findings_summary)
            entries = []  # list of tuples: (scan_type, filename)
            for st in ("secrets", "sast", "sca"):
                for f in self.findings_summary.get(st, {}).get("files", []):
                    entries.append((st, f))

            if not entries:
                self.logger.info("No vulnerable files to display.")
                return
            
            print_list()
            while True:
                cmd = input("Enter command [list/view <file_index>/analyze <file_index>/help/q]: ").strip()
                if not cmd:
                    continue
                low = cmd.lower()
                if low in ("q", "quit", "exit"):
                    self.logger.info("Exiting viewer.")
                    break
                if low == "list":
                    print_list()
                    continue
                if low == "help":
                    self.logger.info("Commands: list | view <file_index> | analyze <file_index> | q")
                    continue

                parts = cmd.split()
                if len(parts) == 2 and parts[0].lower() in ("view", "analyze") and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx < 1 or idx > len(entries):
                        self.logger.error("Invalid index")
                        continue
                    
                    _, filename = entries[idx - 1]
                    
                    if parts[0].lower() == "view":
                        show_findings(filename)
                    else:
                        self.logger.info(f"Analyzing {filename}...")
                        analysis = await self._analyze_single_file(filename)
                        #self.logger.info(f"\nAnalyses:\n")
                        #pprint.pprint(analysis.get("raw_response"))
                    continue

                self.logger.error("Unrecognized command. Type 'help' for options.")
        except KeyboardInterrupt: # for ctrl+c exit
            self.logger.error("\nViewer interrupted.")
        except Exception as e:
            self.logger.error(f"Interactive viewer error: {e}")

    def _write_scan_results_to_file(self, scan_results: Dict) -> bool:
        """
        Write scan results to the local output directory specified in configuration.
        """
        try:
            local_output_dir = get_settings().get("local_output_directory")
            
            if not local_output_dir:
                self.logger.warning("local_output_directory not configured - skipping file output")
                return False
            
            os.makedirs(local_output_dir, exist_ok=True)
            
            scan_type = scan_results.get("pr_metadata", {}).get("scan_type", "unknown")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pr_number = scan_results.get("pr_metadata", {}).get("pr_number", "unknown")
            
            filename = f"security_scan_{scan_type}_{pr_number}_{timestamp}.json"
            file_path = os.path.join(local_output_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(scan_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Scan results written to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write scan results to file: {e}")
            return False

def convert_to_markdown(prediction_data: Dict = None) -> str:
        """
        Convert security analysis prediction data to markdown format
        Note: 
        This function accepts prediction_data in a certain format. 
        Format defined in the "pr_security_review_prompt.toml". AI Analysis
        using _prepare_prediction() will return a string representation of JSON
        """

        if prediction_data is None:
            self.logger.error("prediction data is empty")
            return "# Security Review\n\nNo security analysis available."
        
        # process the prediction data: remove punctuations and convert to dict
        temp = prediction_data
        prediction_data = prediction_data.get("raw_response")
        prediction_data = re.sub(r'```(json)?|\n?', '', prediction_data)
        prediction_data = json.loads(prediction_data)

        md = []
        md.append("# Security Review Report\n")
        
        # Executive Summary
        if prediction_data.get('executive_summary'):
            md.append(f"## Executive Summary\n{prediction_data['executive_summary']}\n")
        
        # Risk Level
        risk_level = prediction_data.get('overall_risk_level', 'unknown').upper()
        md.append(f"## Overall Risk Level: {risk_level}\n")
        
        # Findings Summary
        md.append("## Findings Summary")
        md.append(f"- **Critical:** {prediction_data.get('critical_findings_count', 0)}")
        md.append(f"- **High:** {prediction_data.get('high_findings_count', 0)}")
        md.append(f"- **Medium:** {prediction_data.get('medium_findings_count', 0)}")
        md.append(f"- **Low:** {prediction_data.get('low_findings_count', 0)}\n")
        
        # Detailed Findings
        findings = prediction_data.get('findings_analysis', [])
        if findings:
            md.append("---\n## Detailed Findings\n")
            
            for i, finding in enumerate(findings, 1):
                md.append(f"### {i}. {finding.get('title', 'Untitled Finding')}")
                md.append(f"**Severity:** {finding.get('severity', 'Unknown').upper()} | "
                         f"**Type:** {finding.get('finding_type', 'Unknown')} | "
                         f"**Confidence:** {finding.get('confidence', 'Unknown')}")
                
                location = finding.get('location', {})
                if location:
                    md.append(f"**Location:** `{location.get('file', 'Unknown')}:{location.get('line', 'N/A')}`")
                
                if finding.get('description'):
                    md.append(f"\n**Description:** {finding['description']}")
                
                if finding.get('business_impact'):
                    md.append(f"\n**Business Impact:** {finding['business_impact']}")
                
                if finding.get('attack_scenario'):
                    md.append(f"\n**Attack Scenario:** {finding['attack_scenario']}")
                
                # Immediate Fix
                immediate_fix = finding.get('immediate_fix', {})
                if immediate_fix:
                    md.append("\n**Immediate Fix:**")
                    if immediate_fix.get('action'):
                        md.append(f"- **Action:** {immediate_fix['action']}")
                    if immediate_fix.get('estimated_effort'):
                        md.append(f"- **Effort:** {immediate_fix['estimated_effort']}")
                    if immediate_fix.get('code_example'):
                        md.append(f"\n```\n{immediate_fix['code_example']}\n```")
                
                # Long-term Prevention
                prevention = finding.get('long_term_prevention', [])
                if prevention:
                    md.append("\n**Long-term Prevention:**")
                    for tip in prevention:
                        md.append(f"- {tip}")
                
                # References
                references = finding.get('references', [])
                if references:
                    md.append("\n**References:**")
                    for ref in references:
                        md.append(f"- {ref}")
                
                md.append("\n---\n")
        
        # Remediation Priority
        priorities = prediction_data.get('remediation_priority', [])
        if priorities:
            md.append("## Remediation Priority\n")
            for i, priority in enumerate(priorities, 1):
                md.append(f"{i}. {priority}")
            md.append("")
        
        # Security Recommendations
        recommendations = prediction_data.get('security_recommendations', {})
        if recommendations:
            md.append("## Security Recommendations\n")
            
            if recommendations.get('immediate_actions'):
                md.append("### Immediate Actions (24-48 hours)")
                for action in recommendations['immediate_actions']:
                    md.append(f"- {action}")
                md.append("")
            
            if recommendations.get('short_term_improvements'):
                md.append("### Short-term Improvements (1-2 weeks)")
                for improvement in recommendations['short_term_improvements']:
                    md.append(f"- {improvement}")
                md.append("")
            
            if recommendations.get('long_term_strategy'):
                md.append("### Long-term Strategy")
                for strategy in recommendations['long_term_strategy']:
                    md.append(f"- {strategy}")
                md.append("")
        
        # False Positive Analysis
        # Although, _extract_findings_summary also performs the de-duplication part on scan_results
        # There's a possibility, that numbers reported by the SCAN tools and actual vulnerability count
        # would be different.  
        false_positives = prediction_data.get('false_positive_analysis', [])
        if false_positives:
            md.append("## False Positive Analysis\n")
            for fp in false_positives:
                md.append(f"- {fp}")
            md.append("")
        
        # Summary
        if prediction_data.get('summary'):
            md.append(f"## Summary\n\n{prediction_data['summary']}\n")
        
        # Footer with metadata
        pr_metadata = temp.get('pr_metadata', {})
        
        md.append("---\n")
        md.append("## Report Metadata")
        if pr_metadata.get('pr_url'):
            md.append(f"- **PR URL:** {pr_metadata['pr_url']}")
        if pr_metadata.get('title'):
            md.append(f"- **PR Title:** {pr_metadata['title']}")
        if pr_metadata.get('branch'):
            md.append(f"- **Branch:** {pr_metadata['branch']}")
        if pr_metadata.get('scan_type'):
            md.append(f"- **Scan Type:** {pr_metadata['scan_type'].upper()}")
        
        md.append(f"- **Generated:** {temp.get("analysis_timestamp")}")
        md.append("\n*Generated by PR-Agent Security Review*")

        # write to a file (temporary output path)
        local_output_dir = "/app/output/"
        os.makedirs(local_output_dir, exist_ok=True)
        
        filename = f"security_scan_{temp.get("analysis_timestamp")}.md" # fix the name
        file_path = os.path.join(local_output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))

        # Convert to HTML and save as separate file
        html_content = convert_markdown_to_html('\n'.join(md))
        html_filename = f"security_scan_{temp.get('analysis_timestamp')}.html"
        html_file_path = os.path.join(local_output_dir, html_filename)
        
        with open(html_file_path, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)
        
        get_logger().info(f"AI analysis written to: {file_path}")
        get_logger().info(f"HTML markdown format written to: {html_file_path}")
        return '\n'.join(md)

def convert_markdown_to_html(markdown_content: str) -> str:
    """
    Convert markdown content to HTML format
    This method should be replaced with another tool in future. 
    Tools that are focused only on conversion from md to html
    As of now, this is added just as another feature.
    """
    if not markdown_content:
        return ""
    
    try:
        # Try to use markdown library if available
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'toc']
        )
        return html_content
    except ImportError:
        # Fallback to basic markdown-to-HTML conversion
        return _basic_markdown_to_html(markdown_content)

def _basic_markdown_to_html(md_content: str) -> str:
    """
    Basic markdown to HTML conversion without external dependencies
    
    Args:
        md_content (str): Markdown content
        
    Returns:
        str: Basic HTML content
    """
    lines = md_content.split('\n')
    html_lines = []
    in_code_block = False
    code_lang = ""
    
    html_lines.append('<!DOCTYPE html>')
    html_lines.append('<html><head><meta charset="UTF-8">')
    html_lines.append('<title>Security Review Report</title>')
    html_lines.append('<style>')
    html_lines.append('body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }')
    html_lines.append('h1, h2, h3 { color: #333; }')
    html_lines.append('h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }')
    html_lines.append('h2 { border-bottom: 1px solid #666; padding-bottom: 5px; }')
    html_lines.append('code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }')
    html_lines.append('pre { background: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }')
    html_lines.append('table { border-collapse: collapse; width: 100%; margin: 10px 0; }')
    html_lines.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
    html_lines.append('th { background-color: #f2f2f2; }')
    html_lines.append('</style></head><body>')
    
    for line in lines:
        if line.startswith('```'):
            if in_code_block:
                html_lines.append('</pre>')
                in_code_block = False
            else:
                code_lang = line[3:].strip()
                html_lines.append(f'<pre><code class="language-{code_lang}">')
                in_code_block = True
            continue
        
        if in_code_block:
            html_lines.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            continue
        
        if line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif '**' in line:
            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f'<p>{line}</p>')
        elif line.startswith('- '):
            if not html_lines or not html_lines[-1].startswith('<ul>'):
                html_lines.append('<ul>')
            list_item = line[2:]
            html_lines.append(f'<li>{list_item}</li>')
        elif re.match(r'^\d+\. ', line):
            if not html_lines or not html_lines[-1].startswith('<ol>'):
                html_lines.append('<ol>')
            html_lines.append(f'<li>{line[line.find(". ") + 2:]}</li>')
        elif line.strip() == '---':
            html_lines.append('<hr>')
        # Code blocks (inline)
        elif '`' in line:
            line = re.sub(r'`([^`]+)`', r'<code>\1</code>', line)
            html_lines.append(f'<p>{line}</p>')
        # paragraphs tags
        elif line.strip():
            html_lines.append(f'<p>{line}</p>')
        else:
            if html_lines and html_lines[-1].startswith('<li>'):
                if '<ul>' in '\n'.join(html_lines[-10:]):
                    html_lines.append('</ul>')
                elif '<ol>' in '\n'.join(html_lines[-10:]):
                    html_lines.append('</ol>')
            html_lines.append('<br>')
    
    # Close open lists tags 
    if html_lines and html_lines[-1].startswith('<li>'):
        if '<ul>' in '\n'.join(html_lines[-10:]):
            html_lines.append('</ul>')
        elif '<ol>' in '\n'.join(html_lines[-10:]):
            html_lines.append('</ol>')
    
    html_lines.append('</body></html>')
    return '\n'.join(html_lines)

def test():
    a = PRSecurityReview(args=["security_review", "sca"], pr_url="https://git.fplabs.tech/fplabs/serverless/-/merge_requests/293/")
    print(str(a))
    #pprint.pprint(a.content_dict)
    return a
