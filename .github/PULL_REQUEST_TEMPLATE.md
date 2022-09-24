1. [ ] Update documentation to reflect any user-facing changes - in this repository.

2. [ ] Make sure that the changes are covered by unit-tests.

3. [ ] Run `cargo fmt` on the changes.

4. [ ] Please put changelog entries **in the description of this Pull Request**
   if knowledge of this change could be valuable to users. No need to put the
   entries to the changelog directly, they will be transferred to the changelog
   file by maintainers right after the Pull Request merge.
   
   Please remove any items from the template below that are not applicable.

5. [ ] Describe in common words what is the purpose of this change, related
   Github Issues, and highlight important implementation aspects.

Changelog:
```markdown
### Public dependency updates
- [some_crate](https://crates.io/crates/some_crate) 1.0
 
### Breaking changes
Changes to `Foo`:
- Renamed to `Bar`.

### Additions
- Support for the `khr_foobar` extension.

### Bugs fixed
- `bar` panics when calling `foo`.
````
