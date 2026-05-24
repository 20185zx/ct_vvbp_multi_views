# AI Research Workflow

本项目使用一套轻量级科研代码协作工作流。默认不要直接开始改代码，先判断当前处于哪种状态，再选择合适的 Skill。

## 核心原则

- 不明确下一步时，先使用 `/research-workflow`。
- 有明确代码实现目标时，先使用 `/task-plan` 生成任务单，再交给 implementer 或低成本模型执行。
- 代码被修改后，尤其是由低成本模型执行后，必须使用 `/task-review` 审查 diff 或工作区状态。
- 跑完实验、得到指标、发现结果变化时，使用 `/experiment-log` 记录实验信息。
- 不要把任务规划、代码审查、实验记录混在一次输出里。
- 不要在没有任务单或审查结论的情况下继续叠加新改动。
- 对科研代码，要优先保持可复现性、实验公平性、baseline 可比性和指标一致性。

## Skill 使用规则

### `/research-workflow`

当我不确定下一步应该做什么时使用。  
它只负责判断当前应该进入哪个流程，不直接实现代码、不写完整实验日志、不做完整代码审查。

典型场景：
- 当前项目没有明确下一步
- 工作区已有未提交修改
- 有实验结果目录但不知道先整理还是继续开发
- 不确定应该用 `/task-plan`、`/task-review` 还是 `/experiment-log`

### `/task-plan`

当我有明确代码修改、功能实现、bug 修复或实验调试目标时使用。  
它只负责生成任务单，不直接改代码。

典型场景：
- "把 sparse_views 从固定值改成从 config 读取"
- "让低成本模型实现某个功能"
- "先规划这个 bug 应该怎么修"
- "把一个模糊想法变成可执行任务"

### `/task-implement`

当 `/task-plan` 已经生成任务单后使用。  
它负责按照任务单或 Implementer Handoff 执行代码修改，不重新设计任务目标。

典型场景：
- `/task-plan` 已经完成
- 我想让模型按任务单执行
- 我不想再手写长 prompt
- 我只需要说“按上面的任务单执行”

使用方式：
`/task-implement 按上面的任务单执行`

执行后必须运行：
`/task-review 原任务：<任务描述>`

### `/task-review`

当代码已经被修改后使用。  
它负责审查当前 git diff、工作区状态、scope creep、硬编码、实验可复现性风险等。

典型场景：
- 低成本模型刚改完代码
- 我担心代码被改乱了
- 当前工作区有未提交修改
- 当前没有明确任务，但需要 working-tree health review

如果没有明确任务目标，可以这样调用：

`/task-review 当前没有明确任务目标，请基于当前 git status 和 git diff 做 working-tree health review，判断当前修改是否需要拆分，哪些结果目录需要归档或忽略，以及下一步最小整理动作。`

### `/experiment-log`

当实验已经运行、产生指标、日志或结果目录后使用。  
它只负责整理实验记录，不实现代码、不跑实验、不提交。

典型场景：
- 已经得到 PSNR、SSIM、MSE、MAE 等指标
- 比较不同 sparse_views、不同参数、不同模型
- 记录 best epoch、final epoch、loss 曲线、实验现象
- 需要判断结果是否值得保留、是否可用于论文或汇报

## 推荐流程

### 情况 1：不知道下一步做什么

先运行：

`/research-workflow <当前情况>`

### 情况 2：想让模型改代码

先运行：

`/task-plan <任务描述>`

然后运行：

`/task-implement 按上面的任务单执行`

执行后运行：

`/task-review 原任务：<任务描述>`

### 情况 3：代码已经被改了，但不确定是否安全

运行：

`/task-review <原始任务描述或当前担忧>`

如果没有明确任务，则运行 working-tree health review。

### 情况 4：跑完实验或得到结果

运行：

`/experiment-log <实验记录、指标、观察现象>`

### 情况 5：同时有代码改动和实验结果

默认先运行：

`/task-review 当前没有明确任务目标，请做 working-tree health review。`

如果实验细节可能遗忘，则可以先运行：

`/experiment-log <临时实验记录>`

然后再运行 `/task-review`。

## Multi-agent Workflow

本项目现在支持两个项目级 subagent：

### `@implementer`

用于在 `/task-plan` 生成任务单后执行代码修改。  
默认遵守 `/task-implement` 的规则。  
只执行已有任务单，不重新规划，不扩展任务，不自动 commit，不删除文件。

使用方式：
`@implementer 按上面的任务单执行`

### `@reviewer`

用于在 implementer 完成后审查当前 diff。  
默认遵守 `/task-review` 的规则。  
只读审查，不修改代码，不提交，不删除文件。

使用方式：
`@reviewer 审查当前改动是否符合原任务`

### 推荐流程

1. 主对话或强模型运行 `/task-plan <任务目标>`
2. 运行 `@implementer 按上面的任务单执行`
3. 运行 `@reviewer 审查当前改动是否符合原任务`
4. 用户根据审查结果决定是否提交
5. 如有实验结果，再运行 `/experiment-log`

不要让多个 agent 并行修改同一工作区。

## 禁止事项

- 不要在没有确认任务边界时直接大范围重构。
- 不要让低成本模型自行决定实验设计。
- 不要在未记录 config、seed、output path、metric protocol 的情况下把实验结果当作正式结论。
- 不要把临时结果目录直接混入版本库。
- 不要把代码正确性问题和项目整理问题混为一谈。
- 不要把单次实验现象夸大成最终结论。

## 项目目录约定

- `cache/fanbeam_geometry/` 是 fan-beam 几何缓存目录。
- `cache/vvbp_patches/` 是 VVBP patch 缓存目录（.pt 文件），由 `src/data/cache_builder.py` 生成和使用。
- 这些目录用于存放预计算的几何索引或中间缓存文件，不属于正式实验结果。
- 这些缓存可以由代码重新生成，因此一般不需要使用 `/experiment-log` 记录。
- 真正的实验结果应以 config 中的 `save_dir` 为准，通常位于 `outputs/` 下，包括 checkpoint、训练日志、逐 epoch 评估 CSV、可视化 PNG 等。
- 审查工作区时，不要仅根据目录名判断结果类型；应先确认目录内容和生成来源。

## 输出语言

默认使用简体中文输出。  
代码、文件路径、命令、变量名、函数名、config key、raw logs、PSNR、SSIM、MSE、MAE 等保持英文原文。
