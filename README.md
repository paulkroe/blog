# blog
Blog for Github Pages using Hugo


Two-repo structure:

This repo (blog) — the Hugo source. You write content here.
paulkroe.github.io — the published static site, linked as a git submodule at a-simple-site/public/.
Workflow to publish a new post:

Write the post — add a markdown file under a-simple-site/content/posts/en/ (that's where your existing posts like naive-bayes.md live).

Build the site — run Hugo from the site directory:


cd a-simple-site && hugo
look at site: hugo server
This generates the static HTML into a-simple-site/public/ (the submodule).

Publish — commit and push the public submodule first (this updates paulkroe.github.io), then commit the outer repo to track the source change:


cd a-simple-site/public && git add -A && git commit -m "publish new post" && git push
cd ../.. && git add -A && git commit -m "add new post" && git push
GitHub Pages serves the paulkroe.github.io repo directly, so pushing the submodule is what makes the post live