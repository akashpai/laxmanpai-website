# Laxman Pai Website

## Project Overview
- Tribute website for Indian artist Laxman Pai (1926–2021), built by his son Akash Pai
- Static HTML/CSS/JS — no framework
- Live at: https://akashpai.github.io/laxmanpai-website/
- Repo: https://github.com/akashpai/laxmanpai-website
- Deployed via GitHub Actions on push to main

## Tech Stack
- Google Fonts: Cormorant Garamond (serif) + Inter (sans)
- CSS custom properties for Goan coastal color palette
- No build step — edit and push directly

## Color Palette (Goan Coastal Theme)
- Navy/ocean blues: #0A2540, #14476B, #1A6B8C
- Teal/lagoon: #2BA8B8, #4FD1C5, #B8E6E1
- Sand/ivory: #F5EFE4, #FBF7EF
- Gold accent: #D4A04F
- Hero gradient: rgba overlays on painting 130_1993.jpg from Wayback Machine

## Key Images (Wayback Machine URLs)
- Painting 1978: https://web.archive.org/web/20190310234920im_/http://www.laxmanpai.com/wp-content/uploads/2018/04/99_1978.jpg
- Painting 1988: https://web.archive.org/web/20190310234920im_/http://www.laxmanpai.com/wp-content/uploads/2018/04/119_1988.jpg
- Painting 1993 (hero bg): https://web.archive.org/web/20190310234920im_/http://www.laxmanpai.com/wp-content/uploads/2018/04/130_1993.jpg
- With Parrikar: https://web.archive.org/web/20190310234920im_/http://www.laxmanpai.com/wp-content/uploads/2018/04/1.jpeg
- With Modi: https://web.archive.org/web/20190310234920im_/http://www.laxmanpai.com/wp-content/uploads/2018/04/2.jpeg
- Padma Bhushan ceremony: https://web.archive.org/web/20190310234920im_/http://www.laxmanpai.com/wp-content/uploads/2018/04/3.jpeg

## Local Images (to be added to img/ folder)
See img/README.md for required filenames (padma-bhushan-medal.jpg, with-modi-family.jpg, etc.)

## Workflow
- Edit locally in VS Code
- Push to GitHub: git add . && git commit -m "message" && git push
- View live after GitHub Actions goes green (~1 min)
- Never try to push from cloud — proxy is restricted to zerobrush/AI-BotExample-1 only

## Sections
1. Hero — painting 130 background, name, awards eyebrow, description
2. About — biography
3. Paintings (family collection) — local img/ files
4. Archive Paintings — 3 Wayback Machine paintings (1978, 1988, 1993)
5. Painting Series — 23 chronological works
6. Artistic Style — style pillars
7. Journey Timeline — career milestones
8. Dignitaries — photos with Parrikar, Modi, President Kovind
9. Awards — Padma Bhushan, Padma Shri, Gomant Vibhushan
10. Legacy — closing section
11. Footer

## Pending
- Add real photos when Akash provides them (family, award ceremonies)
- Hero text opacity/brightness still being tuned (opacity: 0.8 and 0.7 lines need → 1)
