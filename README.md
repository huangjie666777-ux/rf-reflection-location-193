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

## 带限反射定位

`POST /locate` 在复用解析与去嵌入的基础上，对器件端口 1 的 S11 或端口 2 的 S22 做带限反射定位（`/deembed` 接口保留不变）。除三份文件与 `left_swapped`、`right_swapped` 外，还需表单字段：

- `port`：`1`（S11）或 `2`（S22）。
- `velocity_m_per_s`：传播速度 v，须满足 `0 < v <= 299792458`。
- `distance_min_m`、`distance_max_m`：米制搜索区间，非负且严格递增。
- `amplitude_threshold`：正幅值门槛。
- `min_peak_distance_m`：正最小峰距。
- `zero_padding`：补零倍数，取 `1`、`2`、`4`、`8`。

所有参数必须有限。测量须为 4 到 4096 个等间隔频点：间隔取全段平均，相对偏差超过 1e-8 即拒绝，不重采样。变换使用对称 Hann 窗加权，补零到不小于 `N * zero_padding` 的最小二次幂 M，复数 IFFT 按 `M / sum(window)` 缩放并保留起始频率相位；时间 `t_k = k / (M * df)`，距离 `d = v * t / 2`。搜索上界必须小于无模糊距离 `v / (2 * df)`。不外推直流，也不构造实数阶跃。

峰搜索在区间内筛选达门槛的内部局部极大值（平台取较小距离），按幅值从大到小、同值距离从小到大贪心挑选，与已选峰不足最小峰距则舍弃，最终按距离返回；无峰时峰表为空。

返回 ZIP 包含：

- `device.s2p`：校正后器件 S 参数。
- `reflection_trajectory.csv`：复数反射全轨迹（index、time_s、distance_m、real、imag、magnitude）。
- `peaks.json`：峰表。
- `config.json`：定位配置与派生量。
- `diagnostics.json`：带宽、无模糊距离、名义物理分辨率、回嵌误差及说明。

### 局限

- 补零只对变换做插值，不提高由带宽决定的物理分辨率（约 `v / (2 * bandwidth)`）。
- 峰强度是反射强弱指标，不能当作阻抗值。
- 频点不等间隔、区间越界或参数非法时整次请求拒绝。

### 双反射示例

```bash
.venv/bin/python -m examples.generate_reflection_example
.venv/bin/python -m uvicorn rfdeembed193.api:app --host 127.0.0.1 --port 8000

curl -f \
  -F 'measurement=@examples/reflection_measurement.s2p;type=text/plain' \
  -F 'left_fixture=@examples/reflection_left_fixture.s2p;type=text/plain' \
  -F 'right_fixture=@examples/reflection_right_fixture.s2p;type=text/plain' \
  -F 'left_swapped=false' -F 'right_swapped=false' -F 'port=1' \
  -F 'velocity_m_per_s=200000000' \
  -F 'distance_min_m=0' -F 'distance_max_m=4.5' \
  -F 'amplitude_threshold=0.05' -F 'min_peak_distance_m=0.3' \
  -F 'zero_padding=4' \
  http://127.0.0.1:8000/locate -o locate.zip
```

示例器件在 v = 2e8 m/s 下含 1.2 m（幅值 0.30）与 2.8 m（幅值 0.15）两处已知反射，定位结果应复现这两个位置。

## 源码结构（更新）

- `rfdeembed193/reflect.py`：参数校验、等间隔检查、Hann 窗 IFFT 变换与峰挑选。
- `rfdeembed193/delivery.py`：去嵌入与定位两类 ZIP 打包。
