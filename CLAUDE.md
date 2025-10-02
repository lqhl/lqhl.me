# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal blog called "巴别之塔 - Tower of Babel" (lqhl.me), built with Hugo static site generator and hosted on Cloudflare Pages. The blog uses the hugo-bearblog theme and includes utterances for comments.

## Key Commands

**Development:**
```bash
hugo serve -D
```
Starts local development server with draft posts included.

**Build:**
```bash
hugo
```
Generates static site to `docs/` directory (configured via `publishDir` in config.yaml).

## Architecture

**Content Structure:**
- Blog posts are in `content/blog/`
- Posts can be standalone `.md` files or index bundles (directory with `index.md` for posts with images/assets)
- Research page at `content/research/index.md`

**Configuration:**
- `config.yaml` - Main Hugo configuration
  - Base URL: https://lqhl.me/
  - Publish directory: `docs/`
  - Theme: hugo-bearblog (in `themes/` as git submodule)
  - Utterances comment system configured via params

**Customizations:**
- `layouts/partials/custom_head.html` - Google Analytics and MathJax support for mathematical notation
- `layouts/partials/nav.html` - Custom navigation
- `static/css/custom.css` - Custom styling
- `static/publication/` - Publication files
- `static/images/` - Site images including favicon

**Theme:**
- Uses hugo-bearblog theme (git submodule)
- Theme customizations via layouts overrides (partials in `layouts/partials/`)

**Deployment:**
- Static files generated to `docs/` directory
- Hosted on Cloudflare Pages
- Git-based deployment workflow
