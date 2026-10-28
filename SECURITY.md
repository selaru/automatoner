# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: go to the **Security** tab of this repository and choose **Report a vulnerability**.

Please include:

- what you found and where (file, route or command),
- steps to reproduce it, or a short proof of concept,
- the version or commit you tested, and your operating system,
- what you think the impact is.

You can expect an acknowledgement within 10 working days. I will keep you updated while I look into it, and I will tell you when a fix is released. If you would like credit in the release notes, say so in your report.

This is a personal open source project maintained in spare time, so there is no bug bounty and no guaranteed fix date. Fixing real vulnerabilities comes first.

## Supported versions

Only the latest release and the `main` branch receive security fixes.

## What Automatoner protects, and what it does not

Automatoner is built to run on your own computer or on a server you control. Knowing the design helps you judge what counts as a vulnerability.

**By design:**

- The web interface listens on `127.0.0.1` only by default. It rejects requests with a foreign `Host` or `Origin` header, to stop DNS rebinding and cross-site requests.
- On any other address it refuses to start without an access token. If a token exists, it is required for the whole API and for WebSockets.
- API keys and the Hugging Face token are stored in the data folder (`~/.automatoner`, or `%LOCALAPPDATA%\automatoner` on Windows) and are never returned to the browser or through MCP.
- Local-only mode blocks all cloud LLM providers. An administrator can fix it with `AUTOMATONER_LOCAL_ONLY`.
- The MCP server runs over stdio with limits (read-only mode, import folders, job and size caps, LLM switch).
- An audit log records who did what, without keys, prompts or data.

**Not provided:**

- User accounts or roles. Everyone with the access token can do everything, including using the saved provider keys.
- TLS. Put the server behind a VPN or a reverse proxy if it is reachable by others.
- A sandbox. Training and exports run as the user who started the server, so a malicious model or dataset you choose to download is your risk.
- Tamper-proof audit logs. Forward the log to your own logging system if you need that.
- Protection from someone who already has access to your user account or the data folder.

## In scope

- Bypassing the access token, or the `Host` and `Origin` checks.
- Reading or exfiltrating stored keys or tokens.
- Path traversal or arbitrary file read, write or delete through ids, uploads or exports.
- Running commands or code through the web interface, the API, the CLI or MCP in ways the documentation does not describe.
- Getting around MCP limits or local-only mode.
- Secrets or data ending up in the audit log, or in error messages shown to users.

## Out of scope

- Findings that need local access to the machine or to the data folder.
- Running the server on a public address with no TLS in front of it.
- Vulnerabilities in third-party models, datasets or Python packages. Report those upstream, though I am glad to hear about them so I can update dependencies.
- Denial of service from a user who already holds a valid token.
- Missing security headers on the static page with no demonstrated impact.

## Dependencies

Dependabot and CodeQL are enabled on this repository. Dependency and code scanning alerts are handled in the open, since they do not describe an unreported flaw.

Before you commit it:

- Private reporting: the "Report a vulnerability" button only appears once you turn on private vulnerability reporting. It's under Settings, then Code security.
- Email: the address is the one from your commits. Replace it if you'd rather use a different one.
- Response time: the 5-working-days acknowledgement is a commitment you'd be making. Change it if it's not realistic.
