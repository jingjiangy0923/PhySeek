# rwlocal

这里演示：知道原来 RWI 的启动方式后，如何按 PhySeek README 的方式适配并启动同一个模型。

原 RWI 命令：

```bash
bash run_policy.sh 0 r6000 xingchen runqing /home/ubuntu/newproject/checkpoints/xingchen_shujia/ckpts/step-20000-ema.safetensors
```

新 SDK 命令：

```bash
cd PhySeek
bash rwlocal/run_policy.sh 0 r6000 xingchen runqing /home/ubuntu/newproject/checkpoints/xingchen_shujia/ckpts/step-20000-ema.safetensors
```

`run_policy.sh` 内部实际执行的是 SDK README 里的启动方式：

```bash
physeek run rwlocal.rwi_falcon_policy:RwiFalconPolicy \
  --id runqing \
  --sep wss://172.25.1.62:9668 \
  --policy-type falcon \
  --policy-arg robot_type=xingchen \
  --policy-arg checkpoint_path=/home/ubuntu/newproject/checkpoints/xingchen_shujia/ckpts/step-20000-ema.safetensors \
  --policy-arg return_video=true \
  --insecure
```

其中 `rwlocal/rwi_falcon_policy.py` 是按 SDK 接口写的适配类：

- 继承 `BasePolicy`
- 实现 `reset()` / `set_instruction()` / `update_obs()` / `get_action()`
- 内部复用原 RWI Falcon 模型加载和推理
- 由 `physeek run` 负责注册 policy 并连接 SEP
