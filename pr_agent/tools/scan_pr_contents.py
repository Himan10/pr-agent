import re
import asyncio
import json
import os
import shlex
import tempfile
from typing import Dict, Optional
from pr_agent.log import get_logger

SCA_FILES_FORMAT = [
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", # JavaScript/Node.js
    "requirements.txt", "Pipfile", "Pipfile.lock", "pyproject.toml", "setup.py", "poetry.lock", # Python
    "pom.xml", "build.gradle", "build.gradle.kts", "gradle.lockfile", # Java
    "packages.config", "project.json", # .NET
    "Gemfile", "Gemfile.lock", # Ruby
    "composer.json", "composer.lock", # PHP
    "go.mod", "go.sum", # Go
    "Cargo.toml", "Cargo.lock", # Rust
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml", # Docker
    "conanfile.txt", "vcpkg.json" # Other
]

class SecurityScanner:
    """
    Asynchronous security scanner class for PR content analysis.
    Supports secrets detection, SAST, and SCA scanning.
    """
    
    def __init__(self, content_dict: Dict[str, str]):
        """Initialize the security scanner."""

        self.content_dict = content_dict
        self.logger = get_logger()
        
    async def secrets(self) -> Dict:
        """
        Run noseyparker for secrets detection.
        Scan type: scan - scans the PR Content | report - Generate a report
        These two commands need to be executed as the "scan" does not directly generates the output
        but saves in a datastore, which later be picked by "report" command
        Output: Dict containing secrets scan results
        """
        try:
            results = {}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write content to temporary files
                for filename, content in self.content_dict.items():
                    file_path = os.path.join(temp_dir, os.path.basename(filename))
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                # Build noseyparker "scan" command using shlex
                cmd_str = f"noseyparker scan --datastore {os.path.join(shlex.quote(temp_dir), "datastore.np")} --git-history=none --color auto --progress auto {shlex.quote(temp_dir)}"
                
                cmd = shlex.split(cmd_str)
                
                # Run noseyparker asynchronously
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.logger.info("secrets scan completed successfully")
                else:
                    self.logger.error(f"secrets scan failed: {stderr.decode()}")
                    return {"error": stderr.decode(), "returncode": process.returncode}
                
                # Build the noseyparker "report" command in order to return a json report
                cmd_str_1 = f"noseyparker report --datastore {os.path.join(shlex.quote(temp_dir), "datastore.np")} --format json" 
                cmd = shlex.split(cmd_str_1)
                
                # Run noseyparker asynchronously
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    try:
                        raw_results = json.loads(stdout.decode())
                        self.logger.info("secrets report completed successfully")
                        
                        # Convert list of dictionaries to nested dict format
                        if isinstance(raw_results, list):
                            results = {
                                "secrets": raw_results,
                                "total_findings": len(raw_results),
                                "finding_types": list(set(finding.get("rule_name", "unknown") for finding in raw_results))
                            }
                        else:
                            results = raw_results
                            
                    except json.JSONDecodeError:
                        results = {"raw_output": stdout.decode()}
                else:
                    self.logger.error(f"secrets scanning report failed: {stderr.decode()}")
                    results = {"error": stderr.decode(), "returncode": process.returncode}
                    
        except Exception as e:
            self.logger.error(f"Error running noseyparker: {e}")
            results = {"error": str(e)}
            
        return results
    
    async def sast(self, rules: Optional[str] = "p/owasp-top-ten", config_path: Optional[str] = None) -> Dict:
        """
        Run semgrep for static application security testing.  
        Returns: Dict containing SAST scan results
        """
        try:
            results = {}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write content to temporary files
                file_paths = []
                for filename, content in self.content_dict.items():
                    file_path = os.path.join(temp_dir, os.path.basename(filename))
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    file_paths.append(file_path)
                
                # Build semgrep command using shlex
                cmd_str = 'semgrep --json --quiet'
                
                if rules == "auto":
                    cmd_str += ' --config=auto'
                elif rules.startswith('p/'):
                    cmd_str += f' --config {shlex.quote(rules)}'
                else:
                    cmd_str += f' --config {shlex.quote(rules)}'
                
                if config_path:
                    cmd_str += f' --config {shlex.quote(config_path)}'
                
                # Add directory to scan
                cmd_str += f' {shlex.quote(temp_dir)}'
                
                cmd = shlex.split(cmd_str)
                
                # Run semgrep asynchronously
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode in [0, 1]:  # 0 = no findings, 1 = findings found
                    try:
                        results = json.loads(stdout.decode())
                        self.logger.info(f"sast scan completed with {len(results.get('results', []))} findings")
                    except json.JSONDecodeError:
                        results = {"raw_output": stdout.decode()}
                else:
                    self.logger.error(f"sast scan failed: {stderr.decode()}")
                    results = {"error": stderr.decode(), "returncode": process.returncode}
                    
        except Exception as e:
            self.logger.error(f"Error running sast: {e}")
            results = {"error": str(e)}
            
        return results
    
    async def sca(self, output_format: str = "cyclonedx-json") -> Dict:
        """
        Run syft for software composition analysis.
        Returns: Dict containing SCA scan results
        Tools used: syft and grype
        """
        try:
            results = {}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                written_files = []
                for filename, content in self.content_dict.items():
                    basename = os.path.basename(filename)
                    if basename in SCA_FILES_FORMAT:
                        file_path = os.path.join(temp_dir, basename)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        written_files.append(basename)
                    else:
                        self.logger.debug(f"skipping non-sca file: {filename}")
                
                if not written_files:
                    self.logger.warning("No sca files found in PR content")
                    return {"error": "no dependency files found for sca scanning"}
                
                self.logger.info(f"sca files prepared for scanning: {written_files}")

                # Step 1: Generate SBOM using syft
                self.logger.info("generating SBOM...")
                cmd_str = f'syft scan dir:{temp_dir} -o {shlex.quote(output_format)}'
                cmd = shlex.split(cmd_str)
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                stdout_text = stdout.decode('utf-8', errors='ignore')
                stderr_text = stderr.decode('utf-8', errors='ignore')
                
                if process.returncode != 0:
                    self.logger.error(f"sca scan failed with return code {process.returncode}: {stderr_text}")
                    return {"error": stderr_text or "Unknown sca error", "returncode": process.returncode}
                
                if not stdout_text.strip():
                    self.logger.warning("sca returned empty output")
                    return {"error": "Empty output from sca scan"}
                
                # Parse SBOM
                try:
                    if output_format == "cyclonedx-json":
                        sbom_data = json.loads(stdout_text)
                        components = sbom_data.get('components', [])
                        self.logger.info(f"SBOM generated with {len(components)} components")
                    else:
                        sbom_data = {"raw_output": stdout_text}
                        components = []
                except json.JSONDecodeError as je:
                    self.logger.error(f"failed to parse sca JSON output: {je}")
                    return {"error": stdout_text, "parse_error": str(je)}
                
                # Step 2: Scan for vulnerabilities using grype
                self.logger.info("scanning for vulnerabilities with grype...")
                
                sbom_file = os.path.join(temp_dir, "sbom.json") # require this to pass it to the grype
                with open(sbom_file, 'w') as f:
                    f.write(stdout_text)
                
                # Run grype on the SBOM
                grype_cmd = ['grype', f'sbom:{sbom_file}', '-o', 'json']
                
                grype_process = await asyncio.create_subprocess_exec(
                    *grype_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                grype_stdout, grype_stderr = await grype_process.communicate()
                grype_stdout_text = grype_stdout.decode('utf-8', errors='ignore')
                grype_stderr_text = grype_stderr.decode('utf-8', errors='ignore')
                
                # Parse grype results
                vulnerabilities = []
                if grype_process.returncode == 0:
                    try:
                        grype_data = json.loads(grype_stdout_text)
                        vulnerabilities = grype_data.get('matches', [])
                        self.logger.info(f"found {len(vulnerabilities)} vulnerabilities")
                    except json.JSONDecodeError:
                        self.logger.warning("failed to parse grype output, continuing without vulnerability data")
                        vulnerabilities = []
                else:
                    self.logger.warning(f"sca vulnerability scan failed with return code {grype_process.returncode}: {grype_stderr_text}")
                
                # Combine results (no counting - that's done in extract_findings_summary)
                results = {
                    "sbom": sbom_data,
                    "vulnerabilities": vulnerabilities
                }
                
                self.logger.info("sca scan completed successfully")
                return results
                    
        except Exception as e:
            self.logger.error(f"error in generating SBOM (with components and vulnerabilities): {e}")
            return {"error": str(e)}

    # async def run_all_scans(self, 
    #                        secrets_config: Optional[str] = None,
    #                        sast_rules: str = "auto", 
    #                        sast_config: Optional[str] = None,
    #                        sca_format: str = "json") -> Dict:
    #     """
    #     This method should be used only when there's a need to run multiple scans at once
    #     Functionality to handle this method's output hasn't included in the methods of PRSecurityReview
    #     """
    #     self.logger.info("Starting concurrent security scans...")
        
    #     # Run all scans concurrently
    #     secrets_task = self.secrets(secrets_config)
    #     sast_task = self.sast(sast_rules, sast_config)
    #     sca_task = self.sca(sca_format)
        
    #     secrets_results, sast_results, sca_results = await asyncio.gather(
    #         secrets_task, sast_task, sca_task, return_exceptions=True
    #     )
        
    #     # Handle any exceptions
    #     if isinstance(secrets_results, Exception):
    #         secrets_results = {"error": str(secrets_results)}
    #     if isinstance(sast_results, Exception):
    #         sast_results = {"error": str(sast_results)}
    #     if isinstance(sca_results, Exception):
    #         sca_results = {"error": str(sca_results)}
        
    #     results = {
    #         "secrets": secrets_results,
    #         "sast": sast_results,
    #         "sca": sca_results,
    #         "summary": {
    #             "total_files_scanned": len(self.content_dict),
    #             "scans_completed": 3,
    #             "scans_failed": sum(1 for r in [secrets_results, sast_results, sca_results] 
    #                               if "error" in r)
    #         }
    #     }
        
    #     self.logger.info("All security scans completed")
    #     return results