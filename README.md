# Project cintel-06-custom

## Author: Jordan

Project 6.

---

## Create and activate virtual environment.

Open your project repository in VS Code. 

We'll open a new terminal in VS Code and run a single command to create a new .venv folder for the local project virtual environment.

### Windows Users - Task 1. Create .venv

Run the following command from the project root directory.
 
On Windows, Use PowerShell (not cmd):

```shell
py -m venv .venv
```

### Mac/Linux Users - Task 1. Create .venv

Run the following command from the project root directory.

On Mac/Linux, Use zsh or bash:

```shell
python3 -m venv .venv
```

### Accept VS Code Suggestions

If VS Code asks: We noticed a new environment has been created. 
Do you want to select it for the workspace folder?
Click Yes. 

### Activate and Deactivate Virtual Environment

Run the following command from the project root directory to activate the virtual environment.

```shell
.venv\Scripts\activate
```
Run the following command from the project root directory to deactivate the virtual environment.

```shell
deactivate
```
---

## git-add-commit-push to GitHub Instructions

Instructions to add files to version control, commit changes, and push them to your remote repository.

- git add - stages changes, adds files to source control
- git commit - creates a labeled snapshot of staged changes.
- git push - updates the remote repository

### Before Starting

Ensure your project folder is open in VS Code, and you have made changes (e.g. created the .gitignore and requirements.txt files).

### Task 1. Git add-commit-push

Using a terminal in VS Code (PowerShell, zsh, or bash).

IMPORTANT: 
Replace the commit message with a clear and descriptive note about what you added or changed.
Wrap the commit message in double quotes. 

```shell
git add .
git commit -m "Add .gitignore and requirements.txt files"
git push -u origin main
```

After subsequent changes, you may be able to use a simpler version of the last command:

```shell
git push
``` # cintel-06-custom
Project 6
