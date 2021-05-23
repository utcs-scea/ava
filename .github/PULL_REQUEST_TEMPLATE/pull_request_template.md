## Pull Request checklist
Please, go through these steps before you submit a PR.

1. Make sure that your PR is not a duplicate.
2. If not, then make sure that:

    a. You have done your changes in a separate branch. Branches MUST have descriptive names that start with either the `fix/` or `feature/` prefixes. Good examples are: `fix/cudart_spec_build` or `feature/rdma_channel`.

    b. You have a descriptive commit message with a short title (first line).

    c. You have only one or a small number of commits (if not, squash some into one commit).

    d. Linter doesn't throw any error. If it does, fix them first (e.g. `./scripts/check_cpp_format.sh -fix .`) and amend your commit (`git commit --amend`).

3. **After** these steps, you're ready to open a pull request.

    a. Give a descriptive title to your PR.

    b. Provide a description of your changes.

    c. Put `closes #XXXX` in your comment to auto-close the issue that your PR fixes (if such).

IMPORTANT: Please review the [CONTRIBUTING.md](../CONTRIBUTING.md) file for detailed contributing guidelines.

**PLEASE REMOVE THE ABOVE CHECKLIST BEFORE SUBMITTING**

<!--- Provide a general summary of your changes in the Title above -->

## Description
<!--- Describe your changes in detail -->

## Motivation and Context
<!--- Why is this change required? What problem does it solve? -->
<!--- If it fixes an open issue, please link to the issue here. -->

## How has this been tested?
<!--- Please describe in detail how you tested your changes. -->
<!--- Include details of your testing environment, tests ran to see how -->
<!--- your change affects other areas of the code, etc. -->

## Screenshots (if appropriate):

## Types of changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Document update (this change is mainly a documentation update)

## Checklist:
<!--- Go over all the following points, and put an `x` in all the boxes that apply. -->
<!--- If you're unsure about any of these, don't hesitate to ask. We're here to help! -->
- [ ] My code passes format and lint checks.
- [ ] My change requires a change to the documentation.
- [ ] I have updated the documentation accordingly.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] My changes generate no new warnings.
- [ ] I have tested my code with a reasonable workload.
- [ ] My code may break some other features.
