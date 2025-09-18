#!/bin/bash

# Script to trigger a release workflow for hanzoai (hanzo-engine)

set -e

# Check if version is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v0.6.1"
    exit 1
fi

VERSION=$1

# Validate version format
if ! [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format v*.*.* (e.g., v0.6.1)"
    exit 1
fi

echo "🚀 Triggering release for Hanzo Engine version $VERSION"

# Create and push tag
git tag -a "$VERSION" -m "Release $VERSION - Hanzo Engine"
git push origin "$VERSION"

echo "✅ Release triggered! Check GitHub Actions for build progress:"
echo "   https://github.com/hanzoai/engine/actions"