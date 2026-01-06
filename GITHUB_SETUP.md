# 🚀 Setup Instructions for GitHub

## Create the Repository on GitHub

1. **Go to GitHub**: Visit https://github.com/new
2. **Repository Name**: `yesno-events`
3. **Description**: `Real-time prediction markets for word frequencies in news cycles`
4. **Visibility**: Choose Public or Private
5. **IMPORTANT**: Do NOT initialize with README, .gitignore, or license
6. **Click**: "Create repository"

## Push Your Code

After creating the repository, run these commands on your local machine:

```bash
# Navigate to the project directory
cd ~/yesno-events

# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/yesno-events.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: Using GitHub CLI

If you have `gh` CLI installed:

```bash
# Create repo and push in one command
cd ~/yesno-events
gh repo create yesno-events --public --source=. --remote=origin --push
```

## What's Included

The repository is ready with:
- ✅ All source code
- ✅ Next.js configuration
- ✅ TypeScript setup
- ✅ Tailwind CSS
- ✅ Complete README
- ✅ .gitignore
- ✅ Package.json with metadata

## Repository Structure

```
yesno-events/
├── app/                    # Next.js App Router
├── components/            # React components
├── logic/                # Business logic
├── services/             # News providers
├── store/                # State management
├── types/                # TypeScript types
├── lib/                  # Utilities & constants
├── package.json          # Dependencies
├── tsconfig.json         # TypeScript config
├── tailwind.config.ts    # Tailwind config
└── README.md             # Documentation
```

## After Pushing

1. **Enable GitHub Pages** (optional): Settings → Pages → Source: GitHub Actions
2. **Add Topics**: Click "About" → Add topics: `prediction-markets`, `nextjs`, `typescript`, `real-time`
3. **Update Homepage**: Add your deployment URL when ready

## Deploy to Vercel

Easiest deployment option:

1. Go to https://vercel.com
2. Click "Import Project"
3. Select your GitHub repo `yesno-events`
4. Click "Deploy"
5. Done! Your app will be live at `your-project.vercel.app`

## Environment Variables

No environment variables needed for MVP! It uses synthetic data.

For future RSS integration, add:
```env
RSS_FEED_URL=https://your-rss-feed.com
```

---

**Questions?** Open an issue at https://github.com/YOUR_USERNAME/yesno-events/issues
