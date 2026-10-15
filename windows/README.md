# moe-on-a-desk for Windows (native) + NVIDIA GPU

**ゲーミング PC 1 台(RTX 4090 24 GB+DDR5 128 GB)のネイティブ Windows で、125B の MoE モデルを量子化を落とさずに、実用的な速さで動かす。**
[moe-on-a-desk](../README.md) のネイティブ Windows 版です。WSL2 版([`../wsl/`](../wsl/README.md))の仕組みを Windows のネイティブ
ビルドに移し、この PC で実際に使うときの速さ(1 回きりのコード・文章の応答)を目標にチューニングしました。

対象は **Qwen3.8-Flash-Next 125B(MoE、IQ3_XXS GSQ-RCO、47 GB)**。量子化は未改造の llama.cpp と同じファイルのままで、
速さはすべてコードの変更から来ています。

*Native Windows edition of moe-on-a-desk: Qwen3.8-Flash-Next 125B (IQ3_XXS, unchanged quantization) on one gaming PC
(Core i9-14900K, DDR5 128 GB, RTX 4090 24 GB, Windows 11). Against unmodified llama.cpp 781dbc5 with its own expert cache and MTP
on the same PC and the same files: **+35 %** on average over the four items of the fixed metric (code +45 %, prose +36 %).
One machine tested; see the caveats below.*

> **ご利用にあたっての注意(必ずお読みください)**
> - **自己責任でお使いください。** 本ディレクトリのコード・手順・数値は無保証です。利用によって生じたいかなる損害についても、作者は責任を負いません。
> - **機材に高い負荷がかかります。** GPU をほぼ使い切り、専用 GPU メモリ(VRAM)は 24 GB のうち 約 23 GB を占有します。
>   発熱・消費電力・ファンの騒音が大きくなります。作業中のデータは保存してから、様子を見ながらお試しください。
> - **RAM を大量に固定します。** モデルのエキスパート(約 40 GiB)を RAM にロックし(`-lm mmap+mlock`、OS が退避できなくなります)、
>   サーバーのプロセスは RAM を 約 49 GB 使います。RAM 64 GB 以下の PC では動かないか、ほかのアプリや OS が極端に遅くなります。
> - **モデルは SSD に置いてください。** HDD に置くと読み込みが非常に遅くなります。
> - **llama.cpp を土台にしていますが、改造による副作用は多くの機種では確かめていません。** 動作と数値を確認したのは下記の 1 台だけです。
>   ほかの GPU(RTX 30 / 50 シリーズ、VRAM の小さい GPU)・ドライバ・CPU・メモリ容量では、遅くなる・出力が変わる・動かないといった可能性があります。
>   重要な用途では、未改造の llama.cpp の出力と比べるなどして確認してください。
>
> **Before you use this:** at your own risk, no warranty. It takes about 23 GB of the 24 GB of VRAM and almost all of the GPU, locks about
> 40 GiB of RAM (the server uses about 49 GB of RAM), and needs the model on an SSD. Only one PC was tested (below); other GPUs, drivers,
> CPUs or memory sizes may be slower, give different output or fail. For anything important, compare with unmodified llama.cpp.

---

## 結果(この PC、固定指標)

**測定したモデル(両方の行で同じファイル)**

| | |
|---|---|
| 本体 | **Qwen3.8-Flash-Next 125B**(MoE、エキスパート 512 個から上位 10 個、1 トークンあたり 約 6B が動作)の ISTA-DASLab GSQ-RCO **IQ3_XXS** GGUF、2 分割、約 47 GB(`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`) |
| 下書き(MTP) | 同モデルの MTP ヘッド。比較元は Q8_0 のまま(`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`)、本作はエキスパートだけ Q4_K / Q5_0 にした版(`mtp-Qwen3.8-Flash-Next-Q8e4.gguf`、`tools/make-mtp-q4.bat` で作成。下書きは本体が必ず検証するので出力は変わりません) |

計測は moe-on-a-desk と同じ固定指標です(`tools/abw.py`):思考なし、MTP(モデル自身の下書きヘッド)あり、llama-server の timings の
デコード tok/s。bench = 1K / 4K トークンのプロンプトに 256 トークンの応答を 3 回(`mac125b/tools/measure/` のベンチ)、
コード・文章 = 固定のプロンプト 1 つずつに greedy で 400 トークン。CPU は全 32 スレッドに対する平均使用率、VRAM は GPU 全体の
専用 GPU メモリの使用量(Windows の画面表示など 約 1 GB を含む)です。どちらの行も同じ PC・同じモデルファイル・同じ計測で、各 2 回の平均です。

| 構成 | bench 1K | bench 4K | コード | 文章 | 4 項目の平均 | CPU | RAM | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 比較元:未改造 llama.cpp 781dbc5(内蔵のエキスパートキャッシュ 14 GB、MTP Q8_0 深さ 3) | 115.7 | 116.8 | 67.6 | 68.5 | 92.1 | 9% | 50 GB | 23.4 GB |
| **本作**(`tools/run-win.bat gpu`) | **151.8** | **149.4** | **97.9** | **93.4** | **123.1** | 12% | 49 GB | 23.2 GB |
| 比較元からの伸び | +31% | +28% | **+45%** | **+36%** | **+35%** | | | |

- **4 項目の平均で +35%**(各項目の伸び率の平均。tok/s の平均では 92.1 → 123.1 tok/s(+34%))。1 回きりの応答に近いコード・文章が特に伸びています。
- 出力:コードの答えは未改造と**一字一句同じ**です。文章は 148 文字目から別の言い回しに分かれます(後述の HC の融合で足し算の順序が変わるため。
  どちらも筋の通った物語)。量子化は同じファイルのままです。
- bench は同じプロンプトを 3 回くり返すため、キャッシュが温まって有利に出ます。実際の使い方に近いのはコード・文章です。
- 2 回の差は、本作で 1〜2%、未改造で最大 7%(bench 1K)でした。比較元も本作と同じ日に、下の「再現の手順」のコマンドで測っています。

## 何をしたか

モデル全体(47 GB)は 24 GB の VRAM に入りません。未改造の llama.cpp でも、エキスパートを RAM に置き、使うものだけを VRAM にキャッシュする
仕組み(`--cpu-moe --moe-cache-mib`)と MTP の下書きで 68〜117 tok/s が出ます。これを比較元として固定し、その上に作ったものは次のとおりです。

1. **GPU 側でのキャッシュの引き当て**(`GGML_OP_MOE_FETCH`、`LLAMA_MOE_CACHE_FETCH=1`、WSL2 版から):足りないエキスパートの判断と取り寄せを
   GPU 自身が行い、ページ固定した RAM から直接読みます。1 回の推論が分割のない 1 つの CUDA グラフになります。ネイティブ Windows 向けに、
   時計・環境変数・動的読み込みなどの移植をしました(patch 0059)。
2. **ハイパーコネクションの混合を 1 つの演算に融合**(`GGML_OP_HC_MIX`、patch 0060):このモデルは 1 層に 2 回、4 本の流れを混ぜる処理
   (流れごとの正規化 → 10240→320 と 10240→4 の行列積 → SiLU → 320→10240 → 重み付き平均)があり、多数の小さなカーネルになっていました。
   これを 3 カーネルの 1 演算にまとめ、全部当たりの検証 1 回(4 トークン)が 17.66 → 15.91 ms になりました。`LLAMA_HC_MIX_FUSED=0` で元に戻ります。
3. **小さなバッチの形ごとのグラフの保持**(`LLAMA_GRAPH_CACHE=16`、Mac 版から):下書きの幅が変わるたびに CUDA グラフを作り直さないようにします。
4. **入力のコピーの非同期化**(patch 0065、`GGML_SCHED_ASYNC_INPUTS=0` で元に戻ります):Nsight Systems で 1 ラウンドを追うと、
   GPU が 約 25% の時間 止まっていました。その一因が、推論のたびに位置・マスク・状態の番号などの入力 約 30 個を
   「コピー → 同期」の 1 つずつで送っていたことでした(GPU が待つ時間 1 回 約 0.35 ms)。ストリームの順序に任せた非同期コピーにし、
   次の入力を書く前に 1 回だけ同期します。出力は一字一句同じで、文章が 約 3.6% 速くなりました。
5. **下書き用ヘッド(MTP)のエキスパートを小さく**(`tools/make-mtp-q4.bat`、WSL2 版から):下書き専用のヘッドのエキスパートだけを
   Q8_0 → Q4_K / Q5_0 にして VRAM を 約 1.1 GB 空け、その分をキャッシュに回しました。下書きは必ず本体が検証するので、本体の出力は変わりません。

### 実験的な機能(既定では無効)

- **キャッシュに無いエキスパートを CPU で計算する経路**(`LLAMA_MOE_CPU=1`、patch 0060〜0064):GPU が足りないエキスパートの仕事を
  ページ固定したメモリの箱に書き、CPU の作業スレッド(既定 8、`LLAMA_MOE_CPU_THREADS`)が計算して返します。
  この PC で以前に測った回ではコードが 120〜139 tok/s まで伸びたこともありましたが、公開前に測り直すと再現せず
  (CPU 使用率 約 33%、文章は 52〜65 tok/s に下がる)、CPU の混み具合で大きく揺れるため、上の結果には含めていません。
  `tools/run-win.bat cpu` で試せますが、実験用とお考えください。

## 実験機(この 1 台だけで確認)

| | |
|---|---|
| CPU | Intel Core i9-14900K(24 コア / 32 スレッド) |
| メモリ | DDR5 128 GB(4 枚、4200 MT/s) |
| GPU | NVIDIA GeForce RTX 4090(VRAM 24 GB)、PCIe 4.0 x16 |
| ストレージ | NVMe SSD(モデルを置いたドライブ) |
| OS | Windows 11 Pro(10.0.26200)、ネイティブ(WSL は使わない) |
| ドライバ / ツール | NVIDIA 610.74、CUDA 13.0(nvcc 13.0.88)、MSVC 14.42(Visual Studio 2022)、CMake + Ninja、`CMAKE_CUDA_ARCHITECTURES=89` |
| 電源・クロック | 既定の設定のまま |

ほかの機材では測っていません。RTX 50 シリーズは `build-win.bat <src> 120` でビルドできますが、数値は未確認です。
VRAM の小さい GPU では `--moe-cache-mib` を小さくすると動きますが、速さは大きく下がります(WSL2 版の 12 GB の結果を参考にしてください)。

## 再現の手順

モデル(含まれていません。配布元から入手してください):
- 本体:ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** GGUF、2 分割(`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`)
- 下書き:Qwen3.8-Flash-Next の MTP ヘッドの **Q8_0** GGUF(`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`)。比較元はこれをそのまま使い、
  本作は `tools/make-mtp-q4.bat` で作る `mtp-Qwen3.8-Flash-Next-Q8e4.gguf` を使います。

ビルド(Git for Windows、Visual Studio 2022 の C++、CUDA 13.0、CMake、Ninja が必要です):

```bat
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4
git -c core.autocrlf=false am <moe-on-a-desk>\mac125b\patches\*.patch <moe-on-a-desk>\wsl\patches\*.patch <moe-on-a-desk>\windows\patches\*.patch
<moe-on-a-desk>\windows\tools\build-win.bat %CD% 89
```

パッチは Mac 版(0001〜0055)、WSL2 版(0056〜0058)、ネイティブ Windows 版(0059〜0065)の順にすべて当てます。
llama.cpp の中には長いパス名のファイルがあるので、短いフォルダ(例 `C:\src\llama.cpp`)で作業してください。

下書きヘッドを小さくする(1 回だけ):

```bat
<moe-on-a-desk>\windows\tools\make-mtp-q4.bat build\bin\llama-quantize.exe mtp-Qwen3.8-Flash-Next-Q8_0.gguf mtp-Qwen3.8-Flash-Next-Q8e4.gguf
```

起動(OpenAI 互換のサーバー、`127.0.0.1:8091`、思考なし):

```bat
set BIN=C:\src\llama.cpp\build\bin
set MODEL=E:\models\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
set MTP=E:\models\mtp-Qwen3.8-Flash-Next-Q8e4.gguf
<moe-on-a-desk>\windows\tools\run-win.bat gpu
```

`run-win.bat` の中身は上の表の「本作」の行そのものです(環境変数 `LLAMA_MOE_CACHE_FETCH=1 LLAMA_GRAPH_CACHE=16`、
`--cpu-moe --moe-cache-mib 14600 -lm mmap+mlock -otd token_embd\.weight=CPU --spec-type draft-mtp --spec-draft-n-max 3`)。
比較元は未改造の 781dbc5 を同じ手順(パッチなし)でビルドし、`run-win.bat base`(`--cpu-moe --moe-cache-mib 14000 -lm mmap+mlock`、MTP Q8_0 深さ 3)で起動します。

計測:`python tools\abw.py NAME -- <サーバーのコマンドライン>`(サーバーを起動して固定指標を流し、CPU・RAM・VRAM も記録します。`psutil` が必要です)。

### VRAM の量を変える

VRAM の使い道は「固定分 約 8.5 GB(本体のエキスパート以外の重み、下書きヘッド、KV、計算用バッファ、画面表示)」+「エキスパートのキャッシュ」です。
`--moe-cache-mib` でキャッシュの大きさ(= VRAM の使用量)を変えられます。満杯まで使うと Windows 側の退避が起きて大きく遅くなるので、
1〜1.5 GB は残してください。

## 出力の品質の確認(正直なところ、軽くしか確かめていません)

確かめたこと:
- 固定指標のコードの答えが、未改造の llama.cpp と一字一句同じ。
- 入力の非同期化(patch 0065)の前後で、コード・文章の答えが一字一句同じ。

まだ確かめていないこと:
- 標準のベンチマーク(MMLU 系、HumanEval 系、長文脈、多言語、ツール呼び出し)、思考あり、温度 > 0 のサンプリング、16K を超える文脈。
- HC の融合は足し算の順序を変えるので、文章のように僅差の候補が多い出力は途中から未改造と別の言い回しになります。害はないと考えていますが、
  ベンチマークでは測っていません。

## ほかの環境での結果を募集しています

1 人と AI エージェントで、1 台の PC だけで行った結果です。ほかの GPU・CPU・メモリ容量・ドライバでの数値(`tools/abw.py` の出力、または
llama-server の timings の tok/s を、未改造とこのフォークの両方で)、品質の評価、不具合の報告、改良の PR を、Issue や PR でぜひお寄せください。
