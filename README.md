# rfdeembed193

两端口 Touchstone 1.0 `.s2p` 夹具去嵌入后端。服务严格解析组合测量、左夹具和右夹具网络，以测量频率为共同网格，对夹具实部/虚部分别线性插值，并使用 NumPy 两端口波传输矩阵求解：

```text
T_measurement = T_left @ T_device @ T_right
T_device = inverse(T_left) @ T_measurement @ inverse(T_right)
```

## 输入规则

HTTP 接口为 `POST /deembed`，使用 `multipart/form-data`。

- 文件字段：`measurement`、`left_fixture`、`right_fixture`。
- 端口方向字段：`left_swapped`、`right_swapped`，可取 `true` 或 `false`。
- 仅接受两端口 S 参数、完整选项行、2 到 20000 个严格递增正频点。
- 支持 `Hz`、`kHz`、`MHz`、`GHz` 和 `RI`、`MA`、`DB`；支持 `!` 注释。
- 三份网络参考阻抗必须相同且为正；夹具必须完整覆盖测量频率，不做外推。
- 任一频点对齐网络的 `|S21| <= 1e-12`、传输矩阵条件数 `> 1e10` 或结果非有限时，整次请求拒绝并标明频率。

## 本地运行

```bash
.venv/bin/python -m examples.generate_example
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q rfdeembed193 examples tests
.venv/bin/python -m uvicorn rfdeembed193.api:app --host 127.0.0.1 --port 8000
```

另开终端下载结果：

```bash
curl -f --fail-with-body \
  -F 'measurement=@examples/measurement.s2p;type=text/plain' \
  -F 'left_fixture=@examples/left_fixture.s2p;type=text/plain' \
  -F 'right_fixture=@examples/right_fixture.s2p;type=text/plain' \
  -F 'left_swapped=false' \
  -F 'right_swapped=false' \
  http://127.0.0.1:8000/deembed -o result.zip
unzip -l result.zip
unzip -p result.zip diagnostics.json
```

返回 ZIP 包含：

- `device.s2p`：测量原频率网格、Hz、RI 格式的器件 S 参数。
- `diagnostics.json`：参考阻抗、左右端口方向、三份输入 SHA256、回嵌最大复数绝对误差。

`examples/` 提供非对称器件、不同左右夹具和不同夹具频率覆盖的可复现示例。

## 源码结构

- `rfdeembed193/touchstone.py`：Touchstone 解析、校验和写出。
- `rfdeembed193/deembed.py`：端口翻转、复值插值、S/T 转换和矩阵级联逆解。
- `rfdeembed193/delivery.py`：ZIP 与 JSON 诊断打包。
- `rfdeembed193/api.py`：FastAPI HTTP 接口与错误响应。
