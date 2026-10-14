# moe-on-a-desk for Windows (WSL2 + NVIDIA GPU)

**ゲーミング PC(RTX 4090 24 GB+DDR5)で、125B の MoE モデルを量子化を落とさずに、PC に余力を残したまま速く動かす。**
[moe-on-a-desk](../README.md) の Windows(WSL2)版です。Mac Studio 向けに作った考え方を CUDA に持ち込み、
**Qwen3.8-Flash-Next 125B(IQ3_XXS、47 GB)を、CPU 使用率 約 3%・専用 GPU メモリ(VRAM)12〜22 GB(キャッシュの大きさで選べる)で**動かします。

この文書の「VRAM」は、グラフィックボードに載っているメモリのことで、Windows のタスクマネージャー([パフォーマンス] → [GPU])の
**「専用 GPU メモリ」**に当たります。同じ画面の「共有 GPU メモリ」(GPU が借りて使う普通の RAM)とは別のものです。

*The Windows (WSL2) edition of moe-on-a-desk: Qwen3.8-Flash-Next 125B (IQ3_XXS, 47 GB, unchanged quantization) on a gaming PC with
an RTX 4090, at about 3 % CPU and a VRAM footprint you choose with one option (12-22 GB). Against unmodified llama.cpp with its own
expert cache on the same PC: code +47 %, prose +29 % at 22 GB; +36 % / +28 % at 12 GB. Outputs are identical to the unmodified
build at the same draft depth. One machine tested; see the caveats.*

> **ご利用にあたっての注意(必ずお読みください)**
> - **自己責任でお使いください。** コード・手順・数値は無保証です。利用によって生じたいかなる損害についても、作者は責任を負いません。
> - **メモリを大量に固定します。** 既定の構成では、モデルファイルのうちエキスパートの部分(約 40 GiB)を RAM 上でページ固定します
>   (OS が退避できなくなります)。サーバーのプロセスは RAM を 約 46 GB 使います。WSL に十分なメモリを割り当て(実験機は 112 GB)、
>   ほかの重いアプリと同時に動かす場合はご注意ください。
> - **専用 GPU メモリ(VRAM)は指定した分を占有します。** 画面表示やほかのアプリの分を残してください。VRAM が満杯に近いと Windows 側の退避が起きて大きく遅くなります。
> - **llama.cpp を土台にしていますが、改造による副作用は多くの機種では確かめていません。** 確かめたのは下記の 1 台(RTX 4090、WSL2)だけです。
>   ほかの GPU・ドライバ・ネイティブ Linux / Windows では、遅くなる・出力が変わる・動かないといった可能性があります。
>   重要な用途では、未改造の llama.cpp の出力と比べるなどして確認してください。
>
> **Before you use this:** at your own risk, no warranty. The default setup page-locks about 40 GiB of RAM (the experts of the model
> file) and the server uses about 46 GB of RAM; the VRAM you configure is taken for the whole session. Only one PC was tested
> (RTX 4090, WSL2); other GPUs, drivers or native Linux / Windows may be slower, give different output or fail.

---

## 何をしたか

モデル全体(47 GB)は 24 GB の VRAM に入りません。llama.cpp の標準のやり方は、入りきらない層のエキスパートを CPU で計算すること
(`--n-cpu-moe`)ですが、この PC では 20〜30 tok/s で、CPU も 30% 以上使います。

最近の llama.cpp には、エキスパートを RAM に置いて**よく使うものだけを VRAM にキャッシュする**仕組み(`--cpu-moe --moe-cache-mib`)があり、
これだけで CPU をほとんど使わずに 64〜117 tok/s が出ます。私たちはこれを比較元として固定し、その上に次の 3 つを作りました。
原理は Mac 版と同じで、「全部を GPU に載せず、使う分だけを GPU 側に置き、足りない分をその場で取り寄せる」です。
Mac は CPU と GPU がメモリを共有していますが、PC では PCIe(実効 約 22 GB/s)を越える必要があるので、その道を太く・短くしました。

1. **エキスパートのページ固定**(`LLAMA_MOE_CACHE_PIN`):RAM 上のエキスパート(mmap)を CUDA に登録し、PCIe の転送を
   12.7 → 22.7 GB/s に、しかも CPU が完了を待たない DMA にしました。llama.cpp には登録用の関数がありましたが、使われていませんでした。
2. **GPU 側でのキャッシュの引き当て**(新しい演算 `GGML_OP_MOE_FETCH`、`LLAMA_MOE_CACHE_FETCH=1`):従来は層ごとに
   「GPU を止める → CPU が選ばれたエキスパートを読む → 足りない分を送る → 再開」を 48 回くり返していました(1 層 約 0.2 ms の往復)。
   これを GPU 自身に任せ、**足りないエキスパートは GPU がページ固定した RAM から直接読みます**。1 回の推論が分割のない 1 つの
   CUDA グラフになり、CPU は推論と推論の合間に LRU を更新して空き枠を補充するだけです。Mac 版の「命令列の中の合図」の考え方の CUDA 版です。
3. **下書き用ヘッド(MTP)のエキスパートを小さく**(`tools/make-mtp-q4.sh`):下書き専用のヘッドのエキスパートを Q8_0 → Q4_K / Q5_0 に。
   下書きは必ず本体が検証するので、**本体の出力は変わりません**(md5 で確認)。VRAM が 約 1.1 GB 空きました。

あわせて、Mac 版のパッチを CUDA で使うと約 256 トークン目で落ちる不具合(QSA の省略が CUDA の疎な注意と合わない)を直しました(patch 0056)。

## 結果(この PC、固定指標)

計測方法は moe-on-a-desk と同じです(`tools/abw.sh`:bench-strata 1K/4K、コード・文章 400 トークン greedy、サーバの timings、思考なし)。
CPU は 32 スレッド全体に対する平均使用率、VRAM は GPU 全体の専用 GPU メモリの使用量(Windows の画面表示など 約 1 GB を含む)です。

### VRAM 22 GB(既定)

| 構成 | bench 1K | bench 4K | コード | 文章 | CPU | RAM(RSS) | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 未改造 llama.cpp、34 層分のエキスパートを CPU で計算、MTP 3 | 26.8 | 29.5 | 27.1 | 17.8 | 31% | 32 GB | 23 GB |
| 未改造 llama.cpp、30 層分を CPU、MTP なし | 22.9 | 23.2 | 20.7 | 20.9 | 33% | 26 GB | 23 GB |
| **比較元**:未改造+内蔵キャッシュ 14 GB、MTP 3(MTP 2 / 4 / 5 はこれより遅い) | 116.7 | 115.3 | 64.6 | 63.1 | 3% | 46 GB | 23 GB |
| 1. ページ固定、MTP 4 | 135.5 | 145.9 | 87.6 | 70.5 | 3% | 46 GB | 23 GB |
| 2. +GPU 側の引き当て、MTP 3 | 124.9 | 124.1 | 89.6 | 83.9 | 3% | 46 GB | 23 GB |
| 2. +GPU 側の引き当て、MTP 4 | 133.8 | 136.9 | 94.4 | 80.2 | 3% | 46 GB | 23 GB |
| **3. +小さい MTP、MTP 4(最良)** | **130.8** | **139.5** | **95.3** | **81.1** | **3%** | 46 GB | **22 GB** |
| 比較元比 | +12% | +21% | **+47%** | **+29%** | | | |

- 出力:2. は比較元と同じ下書き深さで**完全一致**(コード・文章とも md5 一致)。3. は 2. と一致。
- 固定指標とは別の 12 プロンプト(英日、コード・説明・翻訳・計算、300 トークン、`tools/trace_prompts.txt`)でも 54〜102 tok/s、平均 約 83(MTP 3)。
  日本語は遅め(55〜70)。
- bench は同じプロンプトを 3 回くり返すため、キャッシュ方式ではキャッシュが温まって有利に出ます。1 回きりのコード・文章を主に見てください。
- 未改造+内蔵キャッシュの他の深さ:MTP 4 48.8 / 100.5 / 54.9 / 47.0、MTP 5 85.3 / 83.4 / 56.3 / 45.4、MTP 2 77.0 / 82.7 / 55.2 / 55.5。

### VRAM を 12 GB に制限した場合

| 構成 | bench 1K | bench 4K | コード | 文章 | CPU | VRAM |
|---|---:|---:|---:|---:|---:|---:|
| 比較元:未改造+内蔵キャッシュ 6.7 GB、MTP なし | 28.7 | 30.3 | 30.7 | 38.0 | 3% | 12 GB |
| 本作:キャッシュ 4.5 GB+MTP 4 | 34.7 | 35.1 | 35.3 | 25.4 | 3% | 13 GB |
| 本作:キャッシュ 4.5 GB+MTP 3 | 37.0 | 37.3 | 39.9 | 33.3 | 3% | 13 GB |
| **本作:キャッシュ 6.7 GB、MTP なし** | **38.7** | **39.3** | **41.6** | **48.7** | **3%** | **12 GB** |
| 比較元比 | +35% | +30% | +36% | +28% | | |

12 GB では下書きなしが最良です(キャッシュが小さいと、下書きで検証の幅が広がった分のエキスパートの取りこぼしが効くため)。

## 専用 GPU メモリ(VRAM)の使用量はキャッシュの大きさで決まる

VRAM の使い道は「固定分」+「エキスパートのキャッシュ」です。

| 内訳(22 GB 構成) | VRAM |
|---|---:|
| 本体のエキスパート以外の重み(注意・GDN・ハイパーコネクション・共有エキスパート・出力ヘッドなど) | 3.66 GB |
| 計算用バッファ | 0.58 GB |
| 下書き用ヘッド(MTP、エキスパートを Q4_K / Q5_0 にした版) | 約 2.3 GB(元の Q8_0 は 3.29 GB) |
| KV キャッシュ(文脈 16K)・再帰状態など | 約 1 GB |
| **エキスパートのキャッシュ**(`--moe-cache-mib`) | **14 GB(自由に変えられる)** |

- **`--moe-cache-mib N` で VRAM の使用量をほぼそのまま決められます。** 固定分は MTP ありで 約 7.5 GB、MTP なしで 約 5.3 GB です。
- キャッシュが大きいほど取りこぼし(RAM から PCIe で読むエキスパート)が減ります。記録したルーティングでの LRU シミュレーション
  (`tools/cachesim.py`、`tools/lrucap2.py`)では、容量 10% で取りこぼしが 13〜15% 減ります。

| キャッシュ | 3.0 GiB | 4.5 | 6.0 | 7.4 | 9.7 | 14.9 |
|---|---:|---:|---:|---:|---:|---:|
| 取りこぼし(エキスパート/層・回、シミュレーション) | 14.2 | 11.2 | 9.1 | 7.5 | 5.6 | 3.1 |

- 目安 [計算、未計測]:VRAM 16 GB 級ならキャッシュ 約 8.5 GB。12 GB では事前の見込み(コード 50 前後)を実測が下回ったので、控えめにお考えください。
- **満杯まで使わないでください。** 実験機では、キャッシュを 14.0 → 15.1 GB に増やすと(VRAM 全体 23 GB)、逆に 130.8 / 139.5 / 95.3 / 81.1 →
  120.3 / 130.9 / 85.6 / 67.0 に下がりました。画面表示やほかのアプリの分として 1.5〜2 GB は残すのが安全です。
- 起動例:`tools/run-wsl.sh`(22 GB)、`tools/run-wsl.sh 12g`(12 GB)、`CACHE_MIB=9000 tools/run-wsl.sh 12g`(任意の大きさ)。

## 実験機(この 1 台だけで確認)

| | |
|---|---|
| CPU | Intel Core i9-14900K(24 コア / 32 スレッド) |
| メモリ | DDR5 128 GB(4 枚、4200 MT/s)。WSL2 には 112 GB を割り当て |
| GPU | NVIDIA GeForce RTX 4090(VRAM 24 GB、約 1 TB/s)、PCIe 4.0 x16 |
| ストレージ | NVMe SSD(モデルは WSL の ext4 上) |
| OS | Windows 11 Pro + WSL2(Ubuntu 24.04.1 LTS、カーネル 6.18.40.1-microsoft-standard-WSL2) |
| ドライバ / CUDA | NVIDIA 610.74 / CUDA 13.0(nvcc 13.0.88)、gcc 13.3、`-DCMAKE_CUDA_ARCHITECTURES=89` |

この PC での基礎測定(`tools/` の小さなプログラム):

| 項目 | 値 |
|---|---:|
| RAM → VRAM、普通のメモリ(mmap)から 1.87 MB ずつ | 14.1 GB/s(`bw.cu`) |
| 同、ページ固定(`cudaHostRegister`)した mmap から | 22.7 GB/s |
| GPU のカーネルがページ固定した RAM を直接読む(全体 10〜20 ブロック) | 21〜22 GB/s(`zcopy2.cu`、ブロックが多すぎると 11〜15) |
| 参考:VRAM の読み出し | 535 GB/s(同じカーネル) |
| CUDA グラフ内の極小カーネル 1 個 / グラフなしの起動 1 個 | 1.1 µs / 5.6 µs(`graphk.cu`) |
| CUDA グラフの起動 1 回(WSL2) | 約 51 µs(Nsight Systems の API 記録) |
| WSL2 でページ固定したホストのアドレスをカーネルがそのまま使えるか | いいえ(`cudaDevAttrCanUseHostPointerForRegisteredMem` = 0、`uva.cu`) |

## ほかの環境の参考値

同じモデル・同じ量子化の数値を、参考として並べます。機材も方式も違うので、順位ではありません。

| 環境 | 方式 | bench 1K | bench 4K | コード | 文章 |
|---|---|---:|---:|---:|---:|
| Mac Studio M4 Max 64 GB、未改造 llama.cpp(MTP 3) | 全部を統合メモリ上の GPU で | 76.8 | 75.3 | 76.6 | 48.3 |
| Mac Studio M4 Max 64 GB、moe-on-a-desk `-14a` | 同上+Metal の改良 | 125.4 | 117.7 | 117.1 | 70.4 |
| Mac Studio M4 Max 64 GB、Strata v0.1.44(64 GB 常駐の修正込み、MTP 2) | 独立したエンジン | 73.5 | 72.3 | 69.5 | 59.3 |
| **この PC(RTX 4090、WSL2)、本作、VRAM 22 GB** | RAM+VRAM キャッシュ+GPU 側の引き当て | 130.8 | 139.5 | 95.3 | 81.1 |
| **この PC、本作、VRAM 12 GB** | 同上、キャッシュ 6.7 GB | 38.7 | 39.3 | 41.6 | 48.7 |

- Strata(独立したエンジン)の README では、RTX 5070(12 GB)+RAM 64 GB で IQ3_XXS が「応答 62 tok/s、プロンプトの読み込み 1,750 tok/s」
  (32K のプロンプト、4K の応答)とされています。私たちは PC では未計測で、計測方法も異なりますが、**12 GB 級の GPU での応答の速さと、
  プロンプトの読み込みの速さは、Strata の公開値のほうがかなり上**です(本作の読み込みは毎秒 180〜270 トークン)。
  Strata は CPU と GPU を組み合わせてよく練られたエンジンで、本作は「llama.cpp のまま、CPU をほとんど使わずに、VRAM の量を選べる」ことを
  重視しています。目的が違い、どちらにも良さがあります。
- Mac 版の詳細は [../README.md](../README.md) と `../mac125b/` を、PC 版の全記録は [DESIGN.md](DESIGN.md) をご覧ください。

## 試す

モデル(同梱していません。配布元から入手してください):
- 本体:ISTA-DASLab の Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** GGUF(2 分割、`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`)
- 下書き:Qwen3.8-Flash-Next の MTP ヘッド(`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`)。小さい版は `tools/make-mtp-q4.sh` で作ります。

ビルド(WSL2 の Ubuntu、CUDA ツールキット導入済み):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4
git am /path/to/moe-on-a-desk/mac125b/patches/*.patch /path/to/moe-on-a-desk/wsl/patches/*.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF
cmake --build build --target llama-server llama-quantize -j 8
```

(並列数を 24 にすると nvcc が一度落ちたので 8 にしています。RTX 4090 以外では `CMAKE_CUDA_ARCHITECTURES` を合わせてください。)

起動(OpenAI 互換、`http://127.0.0.1:8080/v1`。ブラウザで `http://127.0.0.1:8080` を開くとチャット画面):

```bash
LLAMA_MOE_CACHE_FETCH=1 build/bin/llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -ngl 99 --cpu-moe --moe-cache-mib 14000 \
  -md mtp-Qwen3.8-Flash-Next-Q8e4.gguf --spec-type draft-mtp --spec-draft-n-max 4 \
  -c 16384 -np 1 --reasoning off --host 127.0.0.1 --port 8080
```

12 GB にするなら `--moe-cache-mib 6700` にして `-md` 以降の下書きの 3 つを外します(`tools/run-wsl.sh 12g` と同じ)。

計測:`M=<モデルのパス> tools/abw.sh "名前|llama-server のパス|環境変数|追加の引数"`(固定指標+CPU / RAM / VRAM を表示)。

### スイッチ(このフォークの環境変数)

| 変数 | 既定 | 意味 |
|---|---|---|
| `LLAMA_MOE_CACHE_FETCH` | off | GPU 側の引き当て(ページ固定も自動で行う)。**推奨** |
| `LLAMA_MOE_CACHE_PIN` | off | ページ固定だけ(従来の CPU 経由の引き当てのまま) |
| `LLAMA_MOE_FETCH_POOL` | 256 | グループあたりの空き枠の上限(実際は直近の取りこぼしに合わせて自動で決まる) |
| `LLAMA_MOE_CACHE_PREFETCH=K` | off | 次の層の予測先読み。Flash では予測が当たらず逆効果(記録用に残しています) |
| `GGML_CUDA_MMVF_MAX=N` | off | F16 / BF16 の行列積を幅 N までベクトル用カーネルで。速くなるが**出力の文面が変わる**ため既定では使いません |
| `LLAMA_MOE_CACHE_LOG=N`、`LLAMA_MOE_TRACE=path` | off | 当たり率などの記録、ルーティングの記録(シミュレーション用) |
| `LLAMA_UBATCH_TIME`、`GGML_SCHED_PROF`(`_SYNC`)、`GGML_CUDA_OPPROF`、`GGML_CUDA_GPROF`、`GGML_CUDA_MMDBG` | off | 計測用 |
| `GGML_MOE_FETCH_FIXED`、`GGML_MOE_FETCH_NOCOPY`、`GGML_CUDA_SKIP` | off | 計測専用。**わざと出力を壊します** |

## 品質の確認(軽いものです)

- 確認済み:GPU 側の引き当ては、同じ下書き深さの未改造+内蔵キャッシュと出力が完全一致(固定指標のコード・文章)。小さい MTP は本体の出力を変えない(同じ深さで一致)。
  公開したパッチ列(`../mac125b/patches` 55 本+`patches` 3 本)を 781dbc5 に当てたソースは、計測に使ったソースとツリーのハッシュが一致。
- 未確認:標準ベンチマーク、思考あり、温度 > 0、16K を超える文脈、`-np` 2 以上(1 スロットでしか試していません)、複数 GPU。

## 注意事項

- **RAM**:エキスパート 約 40 GiB をページ固定します(登録は 約 0.7 秒)。サーバーの RSS は 約 46 GB。WSL への割り当てが少ない PC では動かないか、
  ほかのアプリが極端に遅くなるおそれがあります(実験機は WSL に 112 GB)。64 GB 以下の PC は未確認です。
- **専用 GPU メモリ(VRAM)**:指定した分を占有します。満杯に近いと大きく遅くなります(上記)。ゲームなど GPU を使うアプリとの同時使用は想定していません。
- **長いプロンプトの読み込みは遅い**:毎秒 180〜270 トークン程度(大きなバッチでは層ごとにエキスパート全体を PCIe で送るため。手を入れていません)。
- **最初の数回・話題が変わった直後は遅い**:キャッシュが温まるまで取りこぼしが多くなります。日本語は英語より遅めです。
- **Mac 版のパッチを CUDA で使う場合は patch 0056 が必要**です(無いと短いプロンプトから 約 256 トークン目で `illegal memory access` で落ちます)。
- **WSL2 + RTX 4090 の 1 台だけ**で確認しました。ネイティブの Linux / Windows、ほかの GPU、ほかのドライバでは未確認です。
- 計測用のスイッチの一部は、わざと出力を壊します(上の表)。

## 次の予定

- 計算側(1 回の検証の約半分):小さな演算の融合など。全部キャッシュに当たっても 4 トークン幅で 約 17 ms かかっており、重みの量から見た下限(4〜5 ms)より大きい。
- 取りこぼし側(残りの半分):PCIe の帯域(約 22 GB/s)が上限で、入れ替え方針の工夫では大きく減らせないことをシミュレーションで確認しました
  (未来を知る最適方針でも半分)。足りないエキスパートの計算だけを CPU に任せる方式(DDR5 は PCIe の約 3 倍の帯域)を検討中です。
  VRAM が小さい構成ほど効く見込みですが、CPU の使用率が上がるため、余力とのかね合いで判断します。

## 謝辞

llama.cpp / ggml(エキスパートキャッシュ `--moe-cache-mib` を含む土台のすべて)、Qwen、ISTA-DASLab、MTP GGUF の作者、
そして Strata・TensorFold・Sushi・MLX をはじめ、ローカル AI の測定や知見を公開しているみなさんに感謝します。

## ライセンス

パッチは llama.cpp(MIT License、© the ggml authors)を改変するものです。それ以外のファイルのライセンスはリポジトリの所有者が定めます。
