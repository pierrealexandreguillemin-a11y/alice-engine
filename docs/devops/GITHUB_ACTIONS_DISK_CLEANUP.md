# GitHub Actions - Disk Cleanup Guide

**Document ID**: DEVOPS-GHA-DISK-001
**Version**: 1.0.0
**Last Updated**: 2026-01-10
**Applicable**: All repos with large dependencies (PyTorch, TensorFlow, AutoGluon, etc.)

## Problem

GitHub-hosted runners have limited disk space:
- **Ubuntu runners**: ~29GB available (of 84GB OS disk)
- **ARM64 runners**: ~14GB available
- Large ML dependencies can fill this quickly:
  - PyTorch: ~2-3GB
  - TensorFlow: ~2GB
  - AutoGluon: ~500MB + dependencies
  - CUDA/cuDNN: ~2-3GB

## Solution: Pre-installation Cleanup

### Quick Solution (Manual)

Add this step **before** `pip install`:

```yaml
- name: Free Disk Space (for ML dependencies)
  run: |
    sudo rm -rf /usr/share/dotnet          # ~6GB - .NET SDK
    sudo rm -rf /usr/local/lib/android     # ~10GB - Android SDK
    sudo rm -rf /opt/ghc                   # ~5GB - Haskell GHC
    sudo rm -rf /opt/hostedtoolcache/CodeQL # ~1GB - CodeQL
    sudo docker image prune --all --force  # ~3GB - Docker images
    df -h
```

**Total space freed**: ~25GB

### Recommended Solution (Action)

Use the community action for more control:

```yaml
- name: Free Disk Space
  uses: jlumbroso/free-disk-space@main
  with:
    tool-cache: false      # Keep Python/Node caches
    android: true          # Remove Android SDK (~14GB)
    dotnet: true           # Remove .NET SDK (~6GB)
    haskell: true          # Remove GHC (~5GB)
    large-packages: true   # Remove large misc packages (~5GB)
    docker-images: true    # Remove Docker images (~3GB)
    swap-storage: false    # Keep swap (needed for large builds)
```

**Reference**: [jlumbroso/free-disk-space](https://github.com/jlumbroso/free-disk-space)

## What Each Removal Does

| Path | Size | Description | Safe to Remove? |
|------|------|-------------|-----------------|
| `/usr/share/dotnet` | ~6GB | .NET SDK, runtime | YES unless using C#/.NET |
| `/usr/local/lib/android` | ~10-14GB | Android SDK, NDK | YES unless building Android apps |
| `/opt/ghc` | ~5GB | Haskell GHC compiler | YES unless using Haskell |
| `/opt/hostedtoolcache/CodeQL` | ~1GB | GitHub CodeQL analysis | YES unless using CodeQL |
| `/opt/hostedtoolcache/Python` | ~500MB | Python versions | NO - needed for Python projects |
| `/opt/hostedtoolcache/node` | ~500MB | Node.js versions | NO - needed for Node projects |
| Docker images | ~3GB | Pre-pulled images | MAYBE - depends on workflow |

## Project-Specific Considerations

### RAG Projects (EmbeddingGemma, MediaPipe)

For RAG projects with embedding models:

```yaml
- name: Free Disk Space
  uses: jlumbroso/free-disk-space@main
  with:
    android: false         # KEEP if using MediaPipe for mobile
    dotnet: true
    haskell: true
    large-packages: true
    docker-images: true
```

**Note**: Keep Android SDK if building for Android with MediaPipe.

### PyTorch/TensorFlow Projects

```yaml
- name: Free Disk Space
  uses: jlumbroso/free-disk-space@main
  with:
    android: true
    dotnet: true
    haskell: true
    large-packages: true
    docker-images: true
    swap-storage: false    # Keep swap for large model loading
```

### AutoGluon Projects

AutoGluon pulls many dependencies. Full cleanup recommended:

```yaml
- name: Free Disk Space
  run: |
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /usr/local/lib/android
    sudo rm -rf /opt/ghc
    sudo rm -rf /opt/hostedtoolcache/CodeQL
    sudo docker image prune --all --force
    df -h
```

## Alternative: ARM64 Runners (2025)

ARM64 runners are 37% cheaper and faster for compatible workloads:

```yaml
jobs:
  build:
    runs-on: ubuntu-24.04-arm  # ARM64 runner
```

**Caveats**:
- Only 14GB disk space
- Some packages may not have ARM builds
- Free for public repos

## Monitoring Disk Usage

Add this step to debug disk issues:

```yaml
- name: Check Disk Space
  run: |
    echo "=== Disk Usage ==="
    df -h
    echo "=== Largest Directories ==="
    du -sh /opt/* 2>/dev/null | sort -rh | head -10
    du -sh /usr/share/* 2>/dev/null | sort -rh | head -10
    du -sh /usr/local/* 2>/dev/null | sort -rh | head -10
```

## References

- [Free Disk Space Action](https://github.com/jlumbroso/free-disk-space)
- [Maximize Build Disk Space](https://github.com/marketplace/actions/maximize-build-disk-space)
- [GitHub Community Discussion #25678](https://github.com/orgs/community/discussions/25678)
- [Mastering Disk Space on GitHub Actions](https://www.geraldonit.com/mastering-disk-space-on-github-actions-runners/)

---

**Applied in**: `.github/workflows/ci.yml` (Alice-Engine)
