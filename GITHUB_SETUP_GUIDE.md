# GitHub Setup Guide

## Quick Setup

Follow these steps to upload your project to GitHub:

### Step 1: Complete Local Git Setup

Open PowerShell in your project directory and run:

```powershell
cd "C:\Users\yaron.shaposhnik\Dropbox\Projects\Teaching\OMG411\2026 Spring A\[4] SC design\[L] Facility location\code"
.\setup_github.ps1
```

Or manually run:

```powershell
git commit -m "Initial commit: Facility Location Optimization App"
```

### Step 2: Create GitHub Repository

1. **Go to GitHub:** https://github.com/new

2. **Fill in repository details:**
   - **Repository name:** `facility-location-optimization`
   - **Description:** `Interactive Streamlit app for facility location optimization problems (LSCP, MCLP, P-Median, P-Center, SPLP)`
   - **Visibility:** Choose Public or Private
   - **Important:** ❌ DO NOT check:
     - "Add a README file"
     - "Add .gitignore"
     - "Choose a license"
     
     (We already have these files locally)

3. **Click "Create repository"**

### Step 3: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Copy and modify these commands:

```powershell
# Add your GitHub repository as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/facility-location-optimization.git

# Ensure you're on the main branch
git branch -M main

# Push your code to GitHub
git push -u origin main
```

**Example:** If your GitHub username is `john-doe`, the command would be:
```powershell
git remote add origin https://github.com/john-doe/facility-location-optimization.git
```

### Step 4: Verify Upload

1. Refresh your GitHub repository page
2. You should see all your files uploaded
3. The README.md will be displayed on the main page

---

## Troubleshooting

### Authentication Issues

If GitHub asks for authentication, you have two options:

#### Option A: Use Personal Access Token (Recommended)
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "Facility Location Project"
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)
7. When git asks for password, paste the token

#### Option B: Use GitHub CLI
```powershell
# Install GitHub CLI if not already installed
winget install --id GitHub.cli

# Authenticate
gh auth login
```

### "Repository already exists" Error

If the remote already exists:
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/facility-location-optimization.git
```

### Check Current Status

```powershell
# View commit history
git log --oneline

# View remote configuration
git remote -v

# View current branch
git branch
```

---

## Future Updates

After the initial push, to update your GitHub repository:

```powershell
# Stage all changes
git add .

# Commit with a message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

---

## Project Files

Your repository includes:

- `facility_location_app.py` - Main Streamlit application
- `facility_location_solver.py` - Optimization solver implementations
- `test_facility_location_solver.py` - Unit tests
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.gitignore` - Files to exclude from Git
- `multi_panel_figure.ipynb` - Jupyter notebook for visualizations

---

## Questions?

- GitHub Docs: https://docs.github.com/en/get-started/quickstart
- Git Basics: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics

