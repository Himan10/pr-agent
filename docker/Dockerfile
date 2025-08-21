FROM python:3.12.10-slim AS base

RUN apt update && apt install --no-install-recommends -y git curl python3 python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ADD pyproject.toml .
ADD requirements.txt .
ADD docs docs
RUN pip install --no-cache-dir . && rm pyproject.toml requirements.txt
ENV PYTHONPATH=/app

FROM base AS github_app
ADD pr_agent pr_agent
CMD ["python", "-m", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "pr_agent/servers/gunicorn_config.py", "--forwarded-allow-ips", "*", "pr_agent.servers.github_app:app"]

FROM base AS bitbucket_app
ADD pr_agent pr_agent
CMD ["python", "pr_agent/servers/bitbucket_app.py"]

FROM base AS bitbucket_server_webhook
ADD pr_agent pr_agent
CMD ["python", "pr_agent/servers/bitbucket_server_webhook.py"]

FROM base AS github_polling
ADD pr_agent pr_agent
CMD ["python", "pr_agent/servers/github_polling.py"]

FROM base AS gitlab_webhook
ADD pr_agent pr_agent
CMD ["python", "pr_agent/servers/gitlab_webhook.py"]

FROM base AS azure_devops_webhook
ADD pr_agent pr_agent
CMD ["python", "pr_agent/servers/azuredevops_server_webhook.py"]

FROM base AS gitea_app
ADD pr_agent pr_agent
CMD ["python", "-m", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "pr_agent/servers/gunicorn_config.py","pr_agent.servers.gitea_app:app"]


FROM base AS test
ADD requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt && rm requirements-dev.txt
ADD pr_agent pr_agent
ADD tests tests

FROM base AS cli
#add sast tools (ref: ~/pr-agent-workspace/Dockerfile)
# Install noseyparker from GitHub releases
RUN NOSEYPARKER_VERSION="v0.24.0" && \
    curl -L "https://github.com/praetorian-inc/noseyparker/releases/download/${NOSEYPARKER_VERSION}/noseyparker-${NOSEYPARKER_VERSION}-x86_64-unknown-linux-musl.tar.gz" -o noseyparker.tar.gz && \
    tar -xzf noseyparker.tar.gz -C /tmp && \
    ln -s /tmp/bin/noseyparker /bin/noseyparker

#TODO: semgrep should be installed in a virtual env and should be run using /env/bin/python
RUN pip3 install semgrep --break-system-packages

#syft and trivy (for sbom generation)
RUN curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
RUN curl -sSfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

#grypte (to find out vulnerabilities in the packages present in SBOM of a project)
RUN curl -sSfL https://get.anchore.io/grype | sh -s -- -b /usr/local/bin

ADD pr_agent pr_agent
ENTRYPOINT ["python", "pr_agent/cli.py"]