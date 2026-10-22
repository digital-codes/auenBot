# Tasks

## 1. Response exposure

- [x] 1.1 In `botCtl.py` `route_handler`, add `"options": ctx["options"]` to both response payloads when the context carries a non-empty option list, and omit the key otherwise; verify with a live curl round: "gibt es da frösche" returns `state` + `options` (five species titles), while "was macht das reh" returns `state` without an `options` key
- [x] 1.2 Verify the seal is unchanged: decrypt the response token (harness secret) and confirm the options inside the token match the plaintext list, and intent/entity/feature remain token-only

## 2. Harness assertions

- [x] 2.1 In `botApiTest.py`, extend the frosch scenario: assert the option-prompt response carries the plaintext `options` field with exactly the five species titles next to the sealed token, matching the decrypted context's options; verify PASS
- [x] 2.2 Verify the full suite stays green: `python botApiTest.py --scenario all` exits 0 with all scenarios PASS (reh, frosch, the 15 entity samples, breakout, follow-up, tampered, foreign, db_crosscheck, restart_resume, restart_no_secret)

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Frontend consumer work (separate repo, `auenlaend`): switch the response gate and request echo from `context` to `state`, render `message.options` — deploy in lockstep with this change to restore option buttons.
