# Deployment Guide for yesno.events

## Deploy to Vercel (Recommended - Free & Easy)

### Option 1: Deploy via Vercel Dashboard (Easiest)

1. **Create a Vercel Account**
   - Go to [vercel.com/signup](https://vercel.com/signup)
   - Sign up with your GitHub account

2. **Import Your Repository**
   - Click "Add New Project" in your Vercel dashboard
   - Import your `yesno-events` repository from GitHub
   - Vercel will auto-detect it's a Next.js app

3. **Deploy**
   - Click "Deploy"
   - Wait ~1-2 minutes for deployment
   - You'll get a live URL like: `https://yesno-events.vercel.app`

4. **Auto-Deploy on Git Push**
   - Every push to your main branch will auto-deploy
   - Pull requests get preview deployments

### Option 2: Deploy via CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   cd /path/to/yesno-events
   vercel
   ```

4. **Follow the prompts:**
   - Set up and deploy? **Yes**
   - Which scope? **Select your account**
   - Link to existing project? **No**
   - Project name? **yesno-events** (or press enter)
   - Directory? **./** (press enter)
   - Override settings? **No** (press enter)

5. **Production Deployment**
   ```bash
   vercel --prod
   ```

## Alternative: Deploy to Netlify

1. **Create Netlify Account**
   - Go to [netlify.com](https://netlify.com)
   - Sign up with GitHub

2. **Import Repository**
   - Click "Add new site" → "Import an existing project"
   - Select your GitHub repo
   - Build settings:
     - **Build command:** `npm run build`
     - **Publish directory:** `.next`
     - **Framework:** Next.js

3. **Deploy**
   - Click "Deploy site"
   - Get your live URL

## Alternative: Deploy to Railway

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `yesno-events`

3. **Configure**
   - Railway auto-detects Next.js
   - Click "Deploy"
   - Get your live URL

## What Gets Deployed

✅ All your components and pages
✅ Real-time news feed with word tracking
✅ Market creation and closing functionality
✅ Professional UI with gradients
✅ Mock news provider (no external API needed)

## Environment Variables

Currently, your app uses mock data and doesn't require any environment variables. If you add real APIs later:

1. In Vercel: Settings → Environment Variables
2. In Netlify: Site settings → Environment variables
3. In Railway: Variables tab

## Post-Deployment

After deployment, your app will be live at a public URL. You can:
- Share the link with anyone
- Test all market functionality
- The mock news provider will generate sample articles
- Everything works exactly like on localhost

## Troubleshooting

**Build fails?**
- Check that all dependencies are in `package.json`
- Verify Node.js version (Vercel uses latest LTS by default)

**App doesn't load?**
- Check deployment logs in the platform dashboard
- Verify `npm run build` works locally first

**Need help?**
- Vercel docs: https://vercel.com/docs
- Netlify docs: https://docs.netlify.com
- Railway docs: https://docs.railway.app

---

**Recommended:** Use Vercel - it's made by the Next.js team and provides the best experience for Next.js apps.
