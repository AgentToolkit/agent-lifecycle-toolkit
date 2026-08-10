# CHANGELOG

<!-- version list -->

## v0.11.0 (2026-08-10)

### Bug Fixes

- Make structured output work with reasoning models and empty responses
  ([#116](https://github.com/AgentToolkit/altk-boost/pull/116),
  [`ff01f86`](https://github.com/AgentToolkit/altk-boost/commit/ff01f863e60a5587840d1d4626aa456cb0de7f0a))

- Remove unnecessary variable assignment in TestSafeParse
  ([#112](https://github.com/AgentToolkit/altk-boost/pull/112),
  [`dc56d0f`](https://github.com/AgentToolkit/altk-boost/commit/dc56d0fb11f8dc4265e849b7b5a35d82b72bcf39))

- Replace deprecated watsonx model ibm/granite-3-3-8b-instruct
  ([#117](https://github.com/AgentToolkit/altk-boost/pull/117),
  [`ae6201b`](https://github.com/AgentToolkit/altk-boost/commit/ae6201b06862888cbfeaa5edb2697631f204a8b8))

- Replace deprecated watsonx model mistralai/mistral-medium-2505
  ([#117](https://github.com/AgentToolkit/altk-boost/pull/117),
  [`ae6201b`](https://github.com/AgentToolkit/altk-boost/commit/ae6201b06862888cbfeaa5edb2697631f204a8b8))

- Specify ValidationError in tests for invalid config modes and thresholds
  ([#112](https://github.com/AgentToolkit/altk-boost/pull/112),
  [`dc56d0f`](https://github.com/AgentToolkit/altk-boost/commit/dc56d0fb11f8dc4265e849b7b5a35d82b72bcf39))

- Update onnxruntime version for py3.10
  ([#112](https://github.com/AgentToolkit/altk-boost/pull/112),
  [`dc56d0f`](https://github.com/AgentToolkit/altk-boost/commit/dc56d0fb11f8dc4265e849b7b5a35d82b72bcf39))

- Update silent review examples ([#112](https://github.com/AgentToolkit/altk-boost/pull/112),
  [`dc56d0f`](https://github.com/AgentToolkit/altk-boost/commit/dc56d0fb11f8dc4265e849b7b5a35d82b72bcf39))

### Documentation

- Add paper link to README ([#113](https://github.com/AgentToolkit/altk-boost/pull/113),
  [`4d5b442`](https://github.com/AgentToolkit/altk-boost/commit/4d5b442b2aa01e645fd315fa575b02aa17d85a12))

### Features

- Add compact-tool-schema mode, runtime pipeline flag, and SPARC actionable recommendations
  ([#112](https://github.com/AgentToolkit/altk-boost/pull/112),
  [`dc56d0f`](https://github.com/AgentToolkit/altk-boost/commit/dc56d0fb11f8dc4265e849b7b5a35d82b72bcf39))

- Format with black ([#112](https://github.com/AgentToolkit/altk-boost/pull/112),
  [`dc56d0f`](https://github.com/AgentToolkit/altk-boost/commit/dc56d0fb11f8dc4265e849b7b5a35d82b72bcf39))

- Ruff format ([#112](https://github.com/AgentToolkit/altk-boost/pull/112),
  [`dc56d0f`](https://github.com/AgentToolkit/altk-boost/commit/dc56d0fb11f8dc4265e849b7b5a35d82b72bcf39))


## v0.10.1 (2026-03-16)

### Bug Fixes

- Bug in sparc type assignment
  ([#108](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/108),
  [`beceaa5`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/beceaa5594cf8a84162c623f454710faef2f4179))

- Toolguard APIs ([#104](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/104),
  [`a9b90ba`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a9b90bafc08af98100bc06313ac3120d0bf48e94))

- Toolguard comply with Python 3.10
  ([#109](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/109),
  [`c34467f`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/c34467f42bb7788ef251b0e5d3f88a8336bfbfc5))

- Toolguard updates
  ([`4672cd2`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/4672cd2ac3e624ecab0814a3b0ca9676c89214e8))

- Update milvus-lite and disable associated tests
  ([#111](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/111),
  [`64f356d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/64f356d22be1e9cfe922ac903948910e8dc4989f))

- Update onnxruntime version for py3.10
  ([#110](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/110),
  [`187a7ac`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/187a7ac9ab8ef45d0b3e109da23df187698a5f35))

- Update silent review examples
  ([#110](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/110),
  [`187a7ac`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/187a7ac9ab8ef45d0b3e109da23df187698a5f35))

### Documentation

- Add ToolGuard component to landing page
  ([#105](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/105),
  [`63e94ea`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/63e94ea7fb731c1a1ddc3c93ff94f034bcacb751))

- Update images ([#97](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/97),
  [`bb2ee3f`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/bb2ee3fea891d811a71389e5d60912ff092ee73e))

- Updates ([#96](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/96),
  [`dd5cec4`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/dd5cec4d560df0003535b47f520973f4242b6806))


## v0.10.0 (2025-12-23)

### Bug Fixes

- Add watsonx_url to test ([#85](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/85),
  [`c44c150`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/c44c150a4a414c0ffbcecb520321f0b8488b041f))

- Change default schema_field to None in generate methods
  ([#95](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/95),
  [`10c48c5`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/10c48c5823cfc97aaf6eb26a2b4d339e54c72f8a))

- Change import in refraction readme
  ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))

- Change test to watsonx provider instead of openai
  ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))

- JSON Schema snippet generation for parameters - need to be without model_dump()
  ([#95](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/95),
  [`10c48c5`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/10c48c5823cfc97aaf6eb26a2b4d339e54c72f8a))

- Langgraph sparc examples ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))

- Linting issues - unnecessary imports
  ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))

- Refraction dependency - when installing altk without refraction sparc and toolguard cannot run as
  well ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))

- Ruff format ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))

### Documentation

- Enhancements ([#70](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/70),
  [`3bbb983`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/3bbb9836a526fad1b5d03f010841b126f985afa4))

- Fix img urls ([#89](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/89),
  [`4c5ac60`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/4c5ac6013764afca6e45b036d8343a13d6d53029))

- Update index.md ([#86](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/86),
  [`13878f3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/13878f301601bbe313b1252395f945a94774e24f))

- Update mkdocs.yml
  ([`b6e2b70`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/b6e2b7077ca0071f035eacca768dd06f111268ef))

- Update README.md ([#86](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/86),
  [`13878f3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/13878f301601bbe313b1252395f945a94774e24f))

- Update summary.md ([#86](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/86),
  [`13878f3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/13878f301601bbe313b1252395f945a94774e24f))

- Update test-case-generation.md
  ([#86](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/86),
  [`13878f3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/13878f301601bbe313b1252395f945a94774e24f))

- Update tool-enrichment.md ([#86](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/86),
  [`13878f3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/13878f301601bbe313b1252395f945a94774e24f))

- Update tool-validation.md ([#86](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/86),
  [`13878f3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/13878f301601bbe313b1252395f945a94774e24f))

- Updated documentation for Toolops components
  ([#81](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/81),
  [`c7a00a9`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/c7a00a94c779d5480deab9e88da264a262f4938e))

- Updated ToolOps components in ALTK website
  ([#86](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/86),
  [`13878f3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/13878f301601bbe313b1252395f945a94774e24f))

### Features

- Add routing toolkit components
  ([#87](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/87),
  [`a5ef913`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a5ef913fee539b0db5f42f00bddb7d78850ad21d))

- Fix dependencies issues, add new metric to sparc and improvements
  ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))

- SPARC improvements - addition of new metric (spec-free) and handling errors
  ([#94](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/94),
  [`a7f669d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a7f669d15bfb971922e862f4817216c444cacf56))


## v0.9.0 (2025-11-27)


## v0.8.1 (2025-11-25)


## v0.8.0 (2025-11-24)

### Chores

- Group dependencies by lifecycle
  ([#72](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/72),
  [`d0d9ec3`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/d0d9ec305552cbbb7bf8f490bc0a81bf29e6433f))

### Breaking Changes

- If you need to use a package you need to install the lifecycle stage it's a part of as an extra.


## v0.7.1 (2025-11-20)

### Bug Fixes

- Update from parse method to create method
  ([#66](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/66),
  [`df2be54`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/df2be548dc240e338e601bad3ac603b37ef84db7))


## v0.7.0 (2025-11-18)

### Chores

- Rename build lifecycle stage to build_time
  ([#69](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/69),
  [`39bb7b8`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/39bb7b81fc3589f99efb7f847f7e298c9e92aaba))


## v0.6.1 (2025-11-18)

### Bug Fixes

- Stop ignoring nested build folders
  ([#68](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/68),
  [`005d114`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/005d1143004ae194e1d9db51571ccf0bfea4fdcc))


## v0.6.0 (2025-11-18)

### Chores

- Move toolops to the build lifecycle stage
  ([#67](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/67),
  [`b88907e`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/b88907ea2c2ba42f49271f45ccf41db77904bfaa))

### Breaking Changes

- Toolops import will break for previous usage


## v0.5.0 (2025-11-18)

### Bug Fixes

- Use custom StrEnum class for python3.10 support
  ([#63](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/63),
  [`f10aa1b`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/f10aa1bde67087c8afc893923f3aa9fc8e07720d))

### Documentation

- Add ToolGuard to READMEs
  ([`515e1bd`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/515e1bd37a8817885de6fc2a5e2c8c38b594a79f))

- Update CONTRIBUTING.md
  ([`d54cba4`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/d54cba4be83128c606d9996d551c6f6a35e28cd5))

- Update README.md
  ([`95fdaf9`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/95fdaf968bb6b65c39589fbc613ff533ff267f80))

- Update README.md
  ([`52bfb24`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/52bfb24d54913887f901f5d44b226b64dcd25998))

### Features

- Add post_request tool enhancement components
  ([#57](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/57),
  [`ddcf7e4`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/ddcf7e4fcff373ed60d7ae56f8bccdd63219b0e2))

- Add ToolGuard component ([#60](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/60),
  [`c8b6d4d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/c8b6d4dc6dcf16963d67ffcfdeab64242dd95abe))


## v0.4.2 (2025-11-12)

### Documentation

- Minor fixes ([#38](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/38),
  [`0085240`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/0085240cc2308297fa489c05ce01a01130c026ee))

- Update integrations. ([#40](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/40),
  [`e2d8440`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/e2d84400c0e37e31f1e8fad7b228bdb76d3649a2))


## v0.4.1 (2025-10-25)

### Documentation

- Add arxiv links and modify index doc to be in line with README
  ([#31](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/31),
  [`fd503c7`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/fd503c7a5bfe2f8bc029fb01e2839a30f285d9ca))

- Add link to github repo ([#33](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/33),
  [`51cea2d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/51cea2d9401984895a18b8899d405eeb93ff6994))

- Fix broken links ([#32](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/32),
  [`25f8150`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/25f81504f8eb416b5cfc54b48fe39801b2253879))

- Make the "Star us" text bigger and clickable.
  ([#37](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/37),
  [`b55376d`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/b55376d7dddf65e23c2d5b10447b12f434099679))

- Policy guard example notebook
  ([#26](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/26),
  [`e28bdc6`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/e28bdc672a9f564fe91e8f233b038137846cf77c))

- Typeset component problems differently in the readme.
  ([#28](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/28),
  [`ac831f1`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/ac831f1990e648cdf111fe723e54e8e40cc4fab3))

- Update documentation design and branding
  ([#36](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/36),
  [`a99494a`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a99494aaf7ce327df1f95598bf757c3c12ab284b))

- Update mkdocs config ([#30](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/30),
  [`a8fa7e9`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/a8fa7e9b32af69a5f4847c2f8601540ff2b5cf74))


## v0.4.0 (2025-10-21)

### Documentation

- Repo-wide documentation and examples update
  ([#20](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/20),
  [`28597d1`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/28597d1bcc0e2ed624da08fa153fefdc66239e39))


## v0.3.0 (2025-10-16)

- No changes. Reconciling package version with existing pypi releases

## v0.2.0 (2025-10-16)

- No changes. Reconciling package version with existing pypi releases

## v0.1.0 (2025-10-16)

### Features

- Adds direct litellm strings to ComponentConfig
  ([#12](https://github.com/AgentToolkit/agent-lifecycle-toolkit/pull/12),
  [`b38abe2`](https://github.com/AgentToolkit/agent-lifecycle-toolkit/commit/b38abe2636f17ff4776f43448b2db94d1a314c54))


## v0.0.0 (2025-10-15)

- Initial Release
