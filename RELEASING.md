# Releasing Call2EDS

## English
1) Ensure `main` is clean and pushed.
2) Tag a version and push the tag:
```
git tag v0.1.9
git push origin v0.1.9
```
3) GitHub Actions will create a release with:
- Source tarball `call2eds-v0.1.9.tar.gz`
- Docker image published to GHCR (`ghcr.io/jbjardine/call2eds:v0.1.9`)
  and a small note file in the release assets.

If a release fails, check the Actions logs and re-tag after fixing.

## Francais
1) Verifier que `main` est propre et pousse.
2) Tagger une version et pousser le tag:
```
git tag v0.1.9
git push origin v0.1.9
```
3) GitHub Actions genere une release avec:
- Tarball source `call2eds-v0.1.9.tar.gz`
- Image Docker publiee sur GHCR (`ghcr.io/jbjardine/call2eds:v0.1.9`)
  et un fichier note dans les assets.

En cas d'echec, verifier les logs Actions puis re-tagger apres correction.
