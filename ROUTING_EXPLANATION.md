# Frontend URL Routing Explanation

## Why localhost:5173 doesn't change?

Your frontend is a **Single Page Application (SPA)** that uses React state-based navigation instead of URL-based routing. This is why the URL stays at `http://localhost:5173` even when you navigate between different pages (Upload, Skills, Quiz, Results, Dashboard).

### Current Implementation

The app currently uses **state-based navigation**:
- Different pages are shown/hidden based on React state variables
- Example: `showQuizPage`, `showSkillsPage`, `showQuizResultPage`, `showSkillDashboard`
- All navigation happens via state changes, not URL changes
- This is **normal and acceptable** for SPAs

### Benefits of Current Approach
- ✅ Simpler implementation (no routing library needed)
- ✅ Fast navigation (no page reloads)
- ✅ Less boilerplate code

### Drawbacks
- ❌ URL doesn't reflect current page
- ❌ Can't bookmark specific pages
- ❌ Browser back/forward buttons don't work between pages
- ❌ Can't share direct links to specific pages

---

## Option 1: Keep Current Approach (Recommended for MVP)

**If you're okay with the URL staying at localhost:5173**, you can keep the current implementation. It works perfectly fine for a prototype/research project.

**No changes needed!**

---

## Option 2: Add React Router (For Production/Public Demo)

If you want proper URLs like:
- `http://localhost:5173/upload`
- `http://localhost:5173/skills`
- `http://localhost:5173/quiz`
- `http://localhost:5173/results`
- `http://localhost:5173/dashboard`

You can install and configure React Router:

```bash
cd frontend-react
npm install react-router-dom
```

Then update the app to use routes. This requires refactoring the navigation logic.

**Note**: For a research project or MVP, this is usually **not necessary**. The current state-based navigation works fine.

---

## Recommendation

**Keep the current approach** unless you specifically need:
1. Shareable URLs to specific pages
2. Browser back/forward button support
3. Bookmarkable pages

For most use cases (especially for research/demo), the current implementation is sufficient.

