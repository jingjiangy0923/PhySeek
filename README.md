# PhySeek

PhySeek 用来把本地 policy 通过 `wss://` 接入 SEP。

SDK 只负责通信：注册 policy、接收 observation、调用 policy、返回 action。

## 安装

```bash
pip install -e .
```

## Mock 测试

在 `PhySeek` 目录下，先启动 fake slave：

```bash
python examples/mock_slave.py \
  --sep wss://127.0.0.1:9668 \
  --insecure-tls
```

再启动 fake policy：

```bash
physeek run examples.mock_policy:MockPolicy \
  --id mock-policy \
  --sep wss://127.0.0.1:9668 \
  --insecure
```

参数说明：

- `--sep`：SEP 地址，只支持 `wss://`。
- `--id`：注册到 SEP 的 policy id。
- `--insecure` / `--insecure-tls`：仅本地自签证书调试时使用。

## Policy 对接 SDK

外部 policy 只需要继承 `BasePolicy`，实现接收 observation 和返回 action：

```python
import numpy as np
from sdk import Action, BasePolicy, Observation, WireConfig


class MyPolicy(BasePolicy):
    def reset(self):
        pass

    def update_obs(self, observation: Observation):
        self.observation = observation

    def get_action(self) -> Action:
        action = np.zeros((8, 14), dtype=np.float32)
        return Action(action)

    def get_policy_spec(self) -> dict:
        return {
            "output_action_space": "eef",
            "output_action_dim": 14,
        }

    def get_wire_config(self) -> WireConfig:
        return WireConfig(image_format="jpeg", jpeg_quality=90)
```

启动时把本地类传给 `physeek run`：

```bash
physeek run my_policy:MyPolicy \
  --id my-policy \
  --sep wss://your-sep-host:9668
```

其中 `my_policy:MyPolicy` 表示 `my_policy.py` 文件里的 `MyPolicy` 类。
`output_action_space` 和 `output_action_dim` 必须和 SEP 里选择的 slave 匹配。
