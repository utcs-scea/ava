# Contributing to AvA <!-- omit in toc -->

## Getting started <!-- omit in toc -->

Before you begin:
- Do you agree to the [license](LICENSE)?
- Check out the [existing issues](https://github.com/utcs-scea/ava/issues) to see if an issue exists already for the change you want to make.

If you spot something new, open an issue using a [template](https://github.com/utcs-scea/ava/issues/new/choose). We'll use the issue to have a conversation about the problem you want to fix.

### Ready to make a change? Fork the repo
Please DO NOT push directly to any branch in the AvA upstream repository unless the commit is simple and you are one of the core maintainers.

You have done your changes in a separate branch. Branches MUST have descriptive names that start with either the `fix/` or `feature/` prefixes. Good examples are: `fix/cudart_spec_build` or `feature/rdma_channel`.

Please assign any issues that you are working on to your account.

### Open a pull request
When you're done making changes and you'd like to propose them for review, use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) to open your PR (pull request).
It is recommended to read the template before you start to write any code.

We integrate a few format and lint checkers for C++, Python and Shell codes. Please ensure your changes do not fail the linter:

```shell
./scripts/check_cpp_format.sh [-fix] .
./scripts/check_python_format.sh [-fix] .
./scripts/check_python_lint.sh .
./scripts/check_shell_format.sh [-fix] .
```

If necessary, you can set up AvA's pre-commit hook by `./scripts/setup_hooks.sh`.

Once you submit your PR, some core AvA maintainers will review it with you. The first thing you're going to want to do is a self review.
After that, we may have questions, check back on your PR to keep up with the conversation.

### Your PR is merged!
Congratulations! The AvA development team sincerely thank you :heart:.

Once your PR is merged, you will be proudly listed as a contributor in the [contributor chart](https://github.com/utcs-scea/ava/graphs/contributors).

## Starting with an issue

We categorize issues into four kinds:
- The [`question` label](https://github.com/utcs-scea/ava/labels/question) is for any non-development questions such as for setting AvA up.
- The [`bug` label](https://github.com/utcs-scea/ava/labels/bug) is for any bug that exists in the current AvA code or dependencies.
- The [`enhancement` label](https://github.com/utcs-scea/ava/labels/enhancement) is for missing features that AvA may be good to have.
- The [`refactor` label](https://github.com/utcs-scea/ava/labels/refactor) is for refactoring or reimplementing existing codebase, or improving the documentations.

An issue may be tagged with the following labels:
- The [`duplicate` label](https://github.com/utcs-scea/ava/labels/duplicate) is for any duplicated questions or issues.
- The [`invalid` label](https://github.com/utcs-scea/ava/labels/invalid) is for any issue that is not really valid.
- The [`wontfix` label](https://github.com/utcs-scea/ava/labels/wontfix) is for issues that we do not plan to fix in the near future.
- The [`help wanted` label](https://github.com/utcs-scea/ava/labels/help%20wanted) is for any issue that one is working on but needs someone else's help.

## Working in the ava/docs directory or ava/gh-pages branch
Please feel free to contribute to any AvA documentation or submit issues for any missing information.

## Reviewing
We try to review every single PR, but due to the time limit, the process can be slow.
It is appreciated to help review or leave comments to the PRs, so that the contributors can receive feedbacks sooner and the AvA maintainers can conduct final reviews faster.

## :question: Support
We are working hard to improve the code quality, feature coverage, system performance and robustness. Unfortunately, AvA is currently maintained by a very small number of developers at their free time.
We are trying to offer assistance as much as possible, but we cannot make any guarantee of the response speed.
We wish to receive helps from the open-source community, research insititutions and industry companies whoever benefit from AvA's techniques to make AvA better and keep it actively maintained.

If you are having trouble or questions about AvA, please create issues.
If you are intereted in contributing to AvA, using AvA, or sponsoring AvA, please contact [Hangchen Yu's AvA mailbox](ava@yuhc.me).
