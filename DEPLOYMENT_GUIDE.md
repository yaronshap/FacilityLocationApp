# Deploying to Streamlit Community Cloud

## Prerequisites

✅ Your code is on GitHub: https://github.com/yaronshap/FacilityLocationApp  
✅ You have a `requirements.txt` file (already included)  
✅ Your main app file is `facility_location_app.py`

## Step-by-Step Deployment

### Step 1: Sign in to Streamlit Community Cloud

1. Go to: **https://share.streamlit.io/**
2. Click **"Sign in"** or **"Sign up"**
3. Choose **"Continue with GitHub"**
4. Authorize Streamlit to access your GitHub account

### Step 2: Deploy Your App

1. Click **"New app"** (or **"Create app"** button)

2. Fill in the deployment settings:
   - **Repository:** `yaronshap/FacilityLocationApp`
   - **Branch:** `main`
   - **Main file path:** `facility_location_app.py`
   - **App URL (optional):** Choose a custom URL or use the auto-generated one
     - Example: `facility-location-optimizer` or `omg411-facility-location`

3. Click **"Deploy!"**

### Step 3: Wait for Deployment

- Streamlit will automatically:
  - Install dependencies from `requirements.txt`
  - Build and deploy your app
  - This usually takes 2-5 minutes

- You'll see a build log showing the progress

### Step 4: Access Your App

Once deployed, your app will be available at:
- **https://[your-app-name].streamlit.app**
- Example: `https://facility-location-optimizer.streamlit.app`

---

## Important: Check Requirements.txt

Make sure your `requirements.txt` includes all necessary packages. Current contents:

```
streamlit
numpy
matplotlib
pandas
scipy
pulp
```

If you need to add or update packages later, just:
1. Update `requirements.txt` locally
2. Commit and push to GitHub
3. Streamlit will automatically redeploy

---

## Managing Your Deployed App

### Viewing App Settings

1. Go to: https://share.streamlit.io/
2. Find your app in the dashboard
3. Click the **"⋮"** menu for options:
   - **Reboot app** - Restart the app
   - **Delete app** - Remove deployment
   - **Settings** - Advanced configuration
   - **Logs** - View runtime logs
   - **Analytics** - See usage statistics

### Automatic Redeployment

Any time you push changes to GitHub, Streamlit will automatically:
- Detect the changes
- Rebuild the app
- Deploy the new version

### Manual Reboot

If your app has issues:
1. Go to your app dashboard
2. Click **"⋮"** → **"Reboot app"**

---

## Troubleshooting

### Issue: "Module not found" Error

**Solution:** Add the missing package to `requirements.txt`

```powershell
# Add package to requirements.txt
echo "package-name" >> requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Add missing dependency"
git push
```

### Issue: App Runs Slowly

**Solutions:**
- Reduce problem size for default parameters
- Add caching with `@st.cache_data` decorator
- Optimize expensive computations
- Consider upgrading to Streamlit Cloud Pro (more resources)

### Issue: App Won't Deploy

**Check:**
1. `facility_location_app.py` is in the root directory ✓
2. `requirements.txt` is in the root directory ✓
3. All imports are listed in `requirements.txt` ✓
4. No syntax errors in your code

### Issue: "Repository is private"

**Solutions:**
1. Make repository public on GitHub, OR
2. Grant Streamlit access to private repos in GitHub settings

---

## Resource Limits (Free Tier)

Streamlit Community Cloud free tier includes:
- **CPU:** 0.078 cores
- **Memory:** 1 GB RAM
- **Resources:** Shared across all your apps
- **Sleep:** Apps sleep after 7 days of inactivity
- **Bandwidth:** Unlimited

**Note:** For larger problems (many facilities/demands), computation might be slower on the free tier.

---

## Advanced Configuration (Optional)

### Secrets Management

If you need to store API keys or sensitive data:

1. In Streamlit Cloud dashboard, go to app settings
2. Click **"Secrets"**
3. Add secrets in TOML format:

```toml
# .streamlit/secrets.toml format
API_KEY = "your-api-key"
DATABASE_URL = "your-database-url"
```

4. Access in your app:
```python
import streamlit as st
api_key = st.secrets["API_KEY"]
```

### Custom Python Version

Create `.streamlit/config.toml` in your repo:

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

---

## Sharing Your App

Once deployed, share your app with:
- **Direct link:** `https://[your-app-name].streamlit.app`
- **QR code:** Available in app settings
- **Embed:** Use iframe on your website

---

## Monitoring

View your app's:
- **Logs:** Real-time application logs
- **Analytics:** Viewer counts and usage patterns
- **Status:** Uptime and performance metrics

All available in the Streamlit Cloud dashboard.

---

## Next Steps After Deployment

1. ✅ Test all features in the deployed version
2. ✅ Share the link with students/colleagues
3. ✅ Monitor analytics to see usage
4. ✅ Iterate based on user feedback

---

## Quick Reference

| Action | Command |
|--------|---------|
| Deploy app | https://share.streamlit.io/ → New app |
| Update app | `git push` (automatic redeploy) |
| View logs | Dashboard → App → Logs |
| Reboot app | Dashboard → App → ⋮ → Reboot |
| Delete app | Dashboard → App → ⋮ → Delete |

---

## Support Resources

- **Streamlit Docs:** https://docs.streamlit.io/deploy/streamlit-community-cloud
- **Forum:** https://discuss.streamlit.io/
- **GitHub Issues:** https://github.com/streamlit/streamlit/issues

---

## Your Deployment Checklist

- [ ] Sign in to Streamlit Community Cloud
- [ ] Connect GitHub account
- [ ] Click "New app"
- [ ] Select repository: `yaronshap/FacilityLocationApp`
- [ ] Set main file: `facility_location_app.py`
- [ ] Click "Deploy"
- [ ] Wait for build to complete
- [ ] Test deployed app
- [ ] Share link with users

**Estimated time:** 5-10 minutes

Good luck with your deployment! 🚀

