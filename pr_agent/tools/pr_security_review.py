import re
import os
import pprint
import datetime
import json
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
from pr_agent.algo.utils import ModelType, get_model
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
        #answer_str, question_str = self._get_user_answers()
        self.pr_description, self.pr_description_files = (
            self.git_provider.get_pr_description(split_changes_walkthrough=True))
        if (self.pr_description_files and get_settings().get("config.is_auto_command", False) and
                get_settings().get("config.enable_ai_metadata", False)):
            add_ai_metadata_to_diff_files(self.git_provider, self.pr_description_files)
            get_logger().debug(f"AI metadata added to the this command")
        else:
            get_settings().set("config.enable_ai_metadata", False)
            get_logger().debug(f"AI metadata is disabled for this command")

        # instantiate the parent class - SecurityScanner
        super().__init__(self.get_new_hunks_with_line_numbers())
        self.finding = {
            "data": None
        }
        self.findings_summary = None

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
            get_logger().error(f"Error extracting added lines: {e}")
            
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
                get_logger().info(f"SKIP processing hunk for file: {file_info.filename}")
                new_hunks[file_info.filename] = file_info.head_file
                continue
            if not file_info.patch:
                continue
                
            # Get the formatted hunks
            get_logger().info(f"processing hunk for file: {file_info.filename}")
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
                get_logger().info(f"PR has no files: {self.pr_url}, skipping review")
                return None
            
            get_logger().info(f"Running {self.scan_type} scan for PR: {self.pr_url}")
            
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
            
            get_logger().info(f"{self.scan_type.upper()} scan completed for PR: {self.pr_url}")
            
            # write output to a file based on user's choice
            user_choice = input("Would you like to save output in a file? [y/n]: ")
            if user_choice.lower() == 'y':
                self._write_scan_results_to_file(results)
            
            # Send the result to openai handler to retrieve details about the findings
            # get_prediction = True, means it will scan the findings_summary
            # if get_prediction = False, it'll analyze vulnerable files found by the scanners
            predicted_result = await self.get_security_review(results, get_prediction=False)
            if predicted_result:
                pprint.pprint(predicted_result)
            # Offer an interactive file viewer for vulnerable files
            try:
                choice = input("Open interactive vulnerability viewer? [y/n]: ")
                if choice.strip().lower() == 'y':
                    # interactive viewer allows viewing and analyzing files repeatedly
                    await self.interactive_vulnerability_viewer()
            except Exception as _:
                pass
            return predicted_result
            
        except ValueError as e:
           get_logger().error(str(e))
           raise e
        except Exception as e:
           get_logger().error(f"{self.scan_type.upper()} scan failed for PR {self.pr_url}: {e}")
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
            self.findings_summary = self._extract_findings_summary(scan_results)
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
                
                return self.prediction
            else:
                get_logger().warning("prediction is currently set to false")
                return None
            
        except Exception as e:
           get_logger().error(f"AI security analysis failed for PR {self.pr_url}: {e}")
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
                scan_data = self.findings_summary
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
            get_logger().info(f"AI security analysis completed for PR: {self.pr_url}")
            
        except Exception as e:
            get_logger().error(f"AI prediction preparation failed for model {model}: {e}")
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
            get_logger().info(f"Deduplicated {len(findings) - len(deduplicated)} duplicate findings")
            
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
            get_logger().warning(f"Unable to fetch head content for {filename}: {e}")
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
                                            "snippet": match.get("snippet", {}).get("matching")
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
                            "severity": finding.get("extra", {}).get("severity")
                        })
            
            elif self.scan_type == "sca" and os.path.basename(filename) in SCA_FILES_FORMAT:
                for vuln in self.scan_results.get("vulnerabilities", []):
                    findings["findings"].append({
                        "type": "sca",
                        "vuln_id": vuln.get("vulnerability", {}).get("id"),
                        "package": vuln.get("artifact", {}).get("name"),
                        "version": vuln.get("artifact", {}).get("version"),
                        "severity": vuln.get("vulnerability", {}).get("severity")
                    })
        
        except Exception as e:
            findings["error"] = str(e)
        
        #print(findings)
        return findings

    def _extract_findings_summary(self, scan_results: Dict) -> Dict:
        """
        Extracts and summarizes scan results with counts and vulnerable filenames.
        """
        summary = {
            "has_findings": False,
            "secrets": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
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
                        severity = finding.get("extra", {}).get("severity", "medium").lower()
                        if severity in summary["sast"]:
                            summary["sast"][severity] += 1
                        else:
                            summary["sast"]["medium"] += 1
                        
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
        
        return summary

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
                get_logger().info("\nVulnerable files:")
                for idx, (st, f) in enumerate(entries, start=1):
                    get_logger().info(f"  [{idx}] ({st}) {f}")
                get_logger().info("")

            def show_findings(filename):
                findings = self._get_file_findings(filename)
                get_logger().info(f"\nFINDINGS FOR {filename}")
                
                if "error" in findings:
                    get_logger().error(f"Error: {findings['error']}")
                    return
                
                if not findings["findings"]:
                    get_logger().info("No findings for this file.")
                    return
                
                for i, f in enumerate(findings["findings"], 1):
                    get_logger().info(f"[{i}] {f.get('type', 'unknown').upper()}")
                    get_logger().info(f)

            #print(self.findings_summary)
            if not getattr(self, "findings_summary", None):
                get_logger().info("No findings summary available. Run a scan first.")
                return

            # Prepare indexed list
            #print(self.findings_summary)
            entries = []  # list of tuples: (scan_type, filename)
            for st in ("secrets", "sast", "sca"):
                for f in self.findings_summary.get(st, {}).get("files", []):
                    entries.append((st, f))

            if not entries:
                get_logger().info("No vulnerable files to display.")
                return
            
            print_list()
            while True:
                cmd = input("Enter command [list/view <file_index>/analyze <file_index>/help/q]: ").strip()
                if not cmd:
                    continue
                low = cmd.lower()
                if low in ("q", "quit", "exit"):
                    get_logger().info("Exiting viewer.")
                    break
                if low == "list":
                    print_list()
                    continue
                if low == "help":
                    get_logger().info("Commands: list | view <file_index> | analyze <file_index> | q")
                    continue

                parts = cmd.split()
                if len(parts) == 2 and parts[0].lower() in ("view", "analyze") and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx < 1 or idx > len(entries):
                        get_logger().error("Invalid index")
                        continue
                    
                    _, filename = entries[idx - 1]
                    
                    if parts[0].lower() == "view":
                        show_findings(filename)
                    else:
                        get_logger().info(f"Analyzing {filename}...")
                        analysis = await self._analyze_single_file(filename)
                        get_logger().info(f"\n{analysis.get('analysis', analysis)}\n")
                    continue

                get_logger().error("Unrecognized command. Type 'help' for options.")
        except KeyboardInterrupt: # for ctrl+c exit
            get_logger().error("\nViewer interrupted.")
        except Exception as e:
            get_logger().error(f"Interactive viewer error: {e}")

    def _write_scan_results_to_file(self, scan_results: Dict) -> bool:
        """
        Write scan results to the local output directory specified in configuration.
        """
        try:
            local_output_dir = get_settings().get("local_output_directory")
            
            if not local_output_dir:
                get_logger().warning("local_output_directory not configured - skipping file output")
                return False
            
            os.makedirs(local_output_dir, exist_ok=True)
            
            scan_type = scan_results.get("pr_metadata", {}).get("scan_type", "unknown")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pr_number = scan_results.get("pr_metadata", {}).get("pr_number", "unknown")
            
            filename = f"security_scan_{scan_type}_{pr_number}_{timestamp}.json"
            file_path = os.path.join(local_output_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(scan_results, f, indent=2, ensure_ascii=False)
            
            get_logger().info(f"Scan results written to: {file_path}")
            return True
            
        except Exception as e:
            get_logger().error(f"Failed to write scan results to file: {e}")
            return False

def test():
    a = PRSecurityReview(args=["security_review", "sca"], pr_url="https://git.fplabs.tech/fplabs/serverless/-/merge_requests/293/")
    print(str(a))
    #pprint.pprint(a.content_dict)
    return a
