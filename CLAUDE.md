Donot read docs/.~lock.LocalizeSORTv2_Project_Tracker.xlsx# and docs/LocalizeSORTv2_Project_Tracker.xlsx unless specifically asked for.

Task tracking has moved to the GitHub Project associated with this repo (project #5, "MapSORT", owner shinobi-wolf). Use `gh project item-list 5 --owner shinobi-wolf` (and `gh issue view <n>` for details) to find the next task, not the xlsx tracker.

Use conda env nerfly

Only edit and execute/lauch scripts when I start chat messages with the prefix "act:". Otherwise, just brainstorm with me. You can read and run read-only commands when I don't start messages with the prefix "act:".

Ignore docs/modules.yaml and docs/schema.yaml

Ignore tests dir. I have decided to pause work on testing for now.

Verbosity level: low. Keep docstrings, comments, and explanations minimal: one-line docstrings at most, comments only where the why is non-obvious, no restating what the code does.

On a shared/public machine (e.g. a university lab rig), `gh auth logout` before ending the session rather than leaving a persistent GitHub login.