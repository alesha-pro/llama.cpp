# План возврата P2P на 4x RTX 3090 через NVIDIA 610.43.02

Дата аудита: 2026-07-10.

Статус: выполнены только диагностика, анализ исходников и тестовая сборка в `/tmp`. Системный драйвер, загруженные модули, загрузчик и пакеты не менялись.

## Короткий вывод

Переход с 595.71.05 на 610.43.02 — это обновление, а не откат. Он не требует менять установленный CUDA Toolkit и не должен ломать существующую сборку llama.cpp на CUDA 12.8: новый драйвер обратно совместим с приложениями CUDA 12.x. Для CUDA 13.3 версия 610.43.02 является штатным Linux-драйвером из релиза NVIDIA.

Рекомендуемый первый вариант — P2P-only commit `3590ded0` из `aikitoria/open-gpu-kernel-modules`, а не HEAD ветки `610.43.02-p2p`. Следующий commit `565e1b53` автоматически включает отдельный экспериментальный fast-path для `cudaHostRegister` на 1G hugepages, меняет pinning и DMA bookkeeping и для нашей цели не нужен.

Риск перехода оценивается как умеренный операционный и низкий необратимый: сервер headless, Secure Boot выключен, старые пакеты доступны, а штатные модули можно вернуть. Главные риски — смешать kernel и userspace разных версий, позволить штатному DKMS затереть патч при следующем обновлении ядра и принять `topo: OK` за доказательство корректности без `simpleP2P`/all-pairs проверки.

## Текущее состояние

| Компонент | Состояние 2026-07-10 |
| --- | --- |
| ОС | Ubuntu 22.04.5 LTS |
| Ядро | `5.15.0-185-generic` |
| Драйвер | open kernel module `595.71.05` |
| Пакет | `nvidia-driver-595-open` + `nvidia-dkms-595-open` |
| GPU | 4x RTX 3090 24 GiB |
| PCIe | PCIe 3.0 x16 max для каждой карты; idle-состояние Gen1 x16 |
| BAR1 | 32768 MiB на каждой из четырёх карт |
| IOMMU | `amd_iommu=on iommu=pt` уже в cmdline |
| Secure Boot | disabled |
| P2P сейчас | `CNS` во всех 12 межкарточных направлениях |
| Топология | каждая GPU напрямую под своим AMD root-port, все пары показываются как `NODE` |
| ACS | на четырёх GPU root-port включены `ReqRedir+` и `CmpltRedir+` |
| CUDA Toolkit | установлены 12.6 и 12.8; `/usr/local/cuda` ведёт на 12.8, системный `/usr/bin/nvcc` пока выбран от 12.6 |
| Доступный 610 | NVIDIA APT repo предлагает `nvidia-open 610.43.02-1ubuntu1` |

P2P-only исходники 610.43.02 были собраны без установки под текущее ядро. Все модули (`nvidia`, `nvidia-uvm`, `nvidia-modeset`, `nvidia-drm`, `nvidia-peermem`) получили версию `610.43.02` и `vermagic=5.15.0-185-generic SMP mod_unload modversions`.

Причина, по которой прежний P2P исчез: ручной кастомный модуль сохранился только в старом дереве ядра `5.15.0-177/kernel/drivers/video`, а новые ядра получили штатные модули в более приоритетном `updates/dkms`. Текущий `modprobe -D nvidia` выбирает `/lib/modules/5.15.0-185-generic/updates/dkms/nvidia.ko`.

## Выбранная стратегия

Сначала установить и проверить полностью согласованный штатный userspace + kernel driver 610.43.02. Затем заменить только kernel-модули текущего ядра на собранные из P2P-only commit и перезагрузиться. Не запускать `install.sh` вслепую: он ставит модули в `kernel/drivers/video`, не регистрирует DKMS и на пакетной системе может проиграть по приоритету штатному `updates/dkms` — именно этот класс проблемы уже произошёл.

### Этап 0. Подготовить обратимый maintenance window

1. Остановить llama-server и другие CUDA-процессы. Сейчас `/dev/nvidia*` открыт `nvtop`, а `nvidia-persistenced` активен; перед работой их тоже остановить.
2. Работать из `tmux`; желательно иметь IPMI/локальную консоль. Сеть от NVIDIA-драйвера не зависит, но смена kernel-модулей всё равно требует нормального аварийного доступа.
3. Сохранить в отдельный каталог:
   - вывод `dpkg-query`, `dkms status`, `nvidia-smi -q`, `nvidia-smi topo -m`;
   - `/etc/default/grub`, `/etc/modprobe.d`, текущий initramfs;
   - checksum всех активных `nvidia*.ko`;
   - точный список установленных NVIDIA-пакетов и их версий.
4. Заранее скачать пакеты 595.71.05 для offline rollback и пакеты 610.43.02 для установки.
5. Не менять ReBAR, IOMMU, ACS и модульные параметры одновременно с первой сменой драйвера. Так каждый эффект останется диагностируемым.

### Этап 1. Поднять штатный 610.43.02 без P2P-патча

1. Ещё раз выполнить симуляцию APT:

   ```bash
   apt-get -s install nvidia-open=610.43.02-1ubuntu1
   ```

2. Установить пакетный open driver 610.43.02 из уже подключённого NVIDIA CUDA repo. Не смешивать APT-пакеты с `.run`-инсталлятором.
3. Перезагрузиться и проверить:
   - `nvidia-smi` и `/proc/driver/nvidia/version` показывают 610.43.02;
   - все четыре GPU видны;
   - BAR1 остаётся 32768 MiB на каждой GPU;
   - CUDA device query и текущий llama.cpp запускаются;
   - в `dmesg` нет Xid, API mismatch и ошибок загрузки firmware.
4. Сделать короткий baseline на штатном 610 тем же бинарником и теми же параметрами. Это отделит влияние новой версии драйвера от влияния P2P.

Локальные Toolkit 12.6/12.8 на этом этапе не обновлять. Строка `CUDA Version` в `nvidia-smi` — максимальный уровень CUDA Driver API, а не версия используемого `nvcc`.

### Этап 2. Установить только P2P-патч

1. Клонировать fork и зафиксировать P2P-only revision:

   ```bash
   git clone https://github.com/aikitoria/open-gpu-kernel-modules.git
   cd open-gpu-kernel-modules
   git checkout 3590ded022c6e855b1ffa0a09edeb508396935e3
   ```

2. Собрать модули под фактически загруженное ядро и проверить их метаданные:

   ```bash
   make modules -j16
   modinfo -F version kernel-open/nvidia.ko
   modinfo -F vermagic kernel-open/nvidia.ko
   ```

3. Сохранить штатные 610-модули из `/lib/modules/$(uname -r)/updates/dkms/` и их checksum.
4. Установить собранные `nvidia{,-uvm,-modeset,-drm,-peermem}.ko` именно в `updates/dkms`, то есть в реально выбираемый `modprobe` каталог. Затем выполнить `depmod` и обновить initramfs текущего ядра.
5. Перезагрузиться. Не пытаться горячо выгружать занятые модули через минимальный `install.sh`: reboot проще и надёжнее.
6. После загрузки проверить, что checksum загруженного с диска модуля совпадает с P2P build, а kernel/userspace имеют одну версию 610.43.02.

Текущий параметр `NVreg_EnableGpuFirmware=0` сначала сохранить без изменения. Дополнительные `ForceP2P`, `RMForceStaticBar1` и похожие registry overrides не добавлять, пока стандартное поведение P2P-only commit не проверено.

### Этап 3. Проверить не только доступность, но и корректность

Порядок acceptance gate:

1. `nvidia-smi topo -p2p r` и `w`: ожидаем `OK` во всех 12 направлениях.
2. `cudaDeviceCanAccessPeer`: ожидаем `1` во всех 12 направлениях.
3. `p2pBandwidthLatencyTest`: сохранить матрицы disabled/enabled bandwidth и latency.
4. `simpleP2P`: обязательно получить `Test passed`. Отдельный all-pairs correctness probe должен выполнить peer-copy и peer-kernel read/write для каждой из 12 направленных пар и сверить данные.
5. Проверить `dmesg` на `NVRM`, `Xid`, IOMMU faults и PCIe AER.
6. Только после прохождения correctness gate запускать LLM.

Это критично: в issue tracker форка есть случаи, когда `topo` показывал `OK` и bandwidth test выглядел нормально, но `simpleP2P` возвращал нули/битые данные. Один только статус `OK` недостаточен.

### Этап 4. Чистый A/B в llama.cpp

Локальный CUDA backend активирует peer access только при наличии `GGML_CUDA_P2P` и после `cudaDeviceCanAccessPeer`; межкарточные копии затем используют `cudaMemcpyPeerAsync`.

На одном и том же patched 610 выполнить два одинаковых прогона:

1. `GGML_CUDA_P2P` не установлен — контрольный путь без использования P2P в llama.cpp.
2. `GGML_CUDA_P2P=1` — прямой peer path.

Сравнивать:

- prompt processing и decode tok/s отдельно;
- p50/p95 времени токена;
- CPU load и host-memory traffic;
- загрузку всех GPU;
- идентичность первых токенов/логитов на фиксированном seed;
- отсутствие Xid/AER/IOMMU ошибок.

Такой A/B отделит реальный выигрыш P2P от возможных изменений производительности самого драйвера 595 -> 610. На PCIe 3.0 x16 ожидать нужно прежде всего устранения host-staging, лишних синхронизаций и снижения межкарточной latency; конкретный end-to-end выигрыш заранее не обещать.

### Этап 5. ACS — только отдельным экспериментом

Сейчас на всех четырёх AMD root-port включён ACS request/completion redirect. Сначала проверить P2P с текущим ACS. Если correctness проходит, но peer bandwidth/latency явно хуже ожидаемого, провести отдельный контролируемый A/B с временным отключением соответствующих ACS bits и немедленной повторной проверкой всех 12 пар.

Не добавлять `pcie_acs_override` в постоянный cmdline до измерения. Отключение ACS и уже используемый `iommu=pt` уменьшают изоляцию DMA, поэтому на таком хосте нельзя считать безопасными недоверенные PCIe-устройства, VFIO-гостей или недоверенный код.

### Этап 6. Защитить результат от следующего обновления

До создания нормального локального DKMS/deb-пакета временно зафиксировать точные версии NVIDIA-пакетов и не загружать новое ядро без заранее собранного P2P-модуля. Это временная мера: надолго замораживать security updates ядра нельзя.

Постоянное решение:

1. Собрать локальный пакет/DKMS source для точной пары `610.43.02-p2p`.
2. Для каждого установленного ядра автоматически собирать пять модулей.
3. Перед reboot gate проверять `modinfo version`, `vermagic`, checksum и фактический путь `modprobe -D nvidia`.
4. При обновлении NVIDIA не применять старый patch к новой версии автоматически: сначала rebase, build и correctness test.

## Rollback

### Если сломан только P2P-патч

1. Вернуть сохранённые штатные 610.43.02 модули в `updates/dkms`.
2. Выполнить `depmod`, обновить initramfs текущего ядра и перезагрузиться.
3. Проверить stock 610 через `nvidia-smi`, CUDA device query и короткий llama smoke test.

### Если проблемен сам 610

1. Установить сохранённый пакетный набор `595.71.05-0ubuntu0.22.04.1` с `--allow-downgrades`, включая matching userspace, DKMS, kernel source, utilities и libraries.
2. Пересобрать штатный DKMS 595 для `5.15.0-185`, выполнить `depmod`, обновить initramfs и перезагрузиться.
3. При необходимости выбрать в GRUB сохранённое ядро `5.15.0-181`; для него ранее уже был штатный DKMS 595.
4. Проверить отсутствие kernel/userspace API mismatch.

Нельзя возвращать только `nvidia.ko` 595 при userspace 610 или наоборот: это создаёт `API mismatch`/CUDA driver mismatch.

## Выполнено перед первой чистой перезагрузкой

Статус на 2026-07-10 11:45 UTC: установка и live-проверки завершены; ожидается только clean reboot в уже подготовленный initramfs.

- Установлен согласованный пакетный stack `610.43.02-1ubuntu1` (userspace, firmware, open kernel source и DKMS). `dpkg --audit` и `apt-get check` проходят.
- Stock 610 был успешно загружен без reboot и проверен отдельно. Все четыре GPU и BAR1 32768 MiB сохранились; штатный driver показывал `GNS` между consumer GPU, как ожидалось.
- P2P-only commit `3590ded022c6e855b1ffa0a09edeb508396935e3` повторно собран под `5.15.0-185-generic` в `/home/alexey/src/open-gpu-kernel-modules-610-p2p`.
- Пять P2P-модулей установлены в `/lib/modules/5.15.0-185-generic/updates/dkms`, подписаны локальным MOK-ключом и загружены live. `modprobe -D nvidia` выбирает именно этот путь.
- Все пять модулей в initramfs побайтно совпадают с установленными P2P-модулями.
- Package versions NVIDIA и kernel meta-packages временно поставлены на hold до создания постоянного локального DKMS/deb-пакета.
- Полный rollback-набор находится в `/home/alexey/driver-backups/nvidia-p2p-610-20260710T111607Z`: исходные 595-модули, stock 610-модули, P2P-модули, kernel/initramfs, конфиги, checksums, APT logs и все 16 offline-пакетов 595.71.05.

### Correctness и P2P transport

- `nvidia-smi topo -p2p r/w`: `OK` во всех 12 направленных парах.
- Официальный CUDA 12.8 `simpleP2P`: `Test passed`.
- Собственный all-pairs тест: 12/12 `cudaMemcpyPeer` и 12/12 remote-kernel read/write прошли с точной проверкой каждого элемента; 0 ошибок.
- `p2pBandwidthLatencyTest`:
  - unidirectional: примерно 5.84-6.01 GB/s без P2P против 13.15-13.18 GB/s с P2P;
  - bidirectional: примерно 7.36-7.54 GB/s против 25.42-25.53 GB/s;
  - GPU latency: примерно 13.35-18.51 us против 1.41-1.59 us.
- Детерминированный 64-token DeepSeek off/on вывод побайтно идентичен, SHA-256 `e22631561a9a227ca3836ae579de7b6814ed8f2d90157c53bdf85fc7c77ab3b1`.
- Xid, PCIe AER и IOMMU faults после загрузки патча отсутствуют.

### Baseline и llama.cpp A/B

Все числа ниже получены одним бинарником, полным 87-GB GGUF, layer split, `-ts 1,1,1,1`, FlashAttention, no-repack, `ubatch=512`, power limit 280 W.

| Режим | pp512 | tg64 |
| --- | ---: | ---: |
| 595 stock, P2P unavailable, среднее двух | 661.39 t/s | 28.61 t/s |
| 610 stock, P2P unavailable | 625.55 / 665.22 t/s | 28.75 / 28.76 t/s |
| 610 patched, `GGML_CUDA_P2P` unset, среднее двух | 658.91 t/s | 27.82 t/s |
| 610 patched, `GGML_CUDA_P2P=1`, среднее двух | 668.99 t/s | 28.70 t/s |

Короткий patched-driver A/B: примерно `+1.53%` pp и `+3.14%` decode. Более устойчивый 256-token decode A/B дал 28.1104 -> 28.3039 t/s (`+0.69%`), а pp512 661.70 -> 668.50 t/s (`+1.03%`). На текущем layer split end-to-end эффект умеренный; основной новый ресурс — резко более быстрый и низколатентный transport для будущего expert/tensor exchange.

### Известное нефатальное сообщение R610 P2P path

При инициализации CUDA-контекста patched driver сначала пытается выделить client shadow fault buffer с huge GPU page и пишет внутренний `NV_ERR_NO_MEMORY`, после чего штатная функция `_kgmmuClientShadowFaultBufferPagesAllocate` явно повторяет allocation с default page size. CUDA-процессы продолжают работу; многократные полные загрузки модели, CUDA samples, all-pairs correctness и текстовый тест завершились успешно. Это не Xid и не фактический VRAM OOM приложения, но после clean reboot сообщение нужно ещё раз подтвердить и отдельно решить, хотим ли мы убрать шум принудительным default-page path.

## Проверено после чистой перезагрузки

Clean reboot выполнен 2026-07-10 в 11:52 UTC, boot ID `a94005fd-9c5c-423b-ac6b-021bfacf901d`. P2P-конфигурация полностью сохранилась.

- Загружен именно кастомный open kernel module `610.43.02`, собранный `alexey@llm-server`; `modinfo -n nvidia` и `modprobe -D nvidia` указывают на `/lib/modules/5.15.0-185-generic/updates/dkms/nvidia.ko`.
- Все пять установленных модулей побайтно совпадают с сохранённым signed P2P-набором. Kernel `5.15.0-185-generic`, `EnableResizableBar=1`, `EnableGpuFirmware=0`; Secure Boot выключен.
- Все четыре RTX 3090 доступны, BAR1 составляет 32768 MiB на каждой. `nvidia-smi topo -p2p r/w` снова показывает `OK` во всех 12 направлениях.
- Собственный all-pairs correctness test после reboot: 12/12 `cudaMemcpyPeer`, remote read/write и exact verification прошли, 0 ошибок. Официальный CUDA `simpleP2P` также завершился `Test passed`.
- Повторный `p2pBandwidthLatencyTest`: 13.15-13.18 GB/s unidirectional, 25.46-25.53 GB/s bidirectional и 1.40-1.60 us GPU latency между картами.
- Полная загрузка того же 87-GB DeepSeek GGUF и llama.cpp smoke с P2P завершились с code 0: pp512 `654.04 t/s`, tg64 `28.27 t/s`. Первый cold load после reboot был ограничен чтением модели с диска, а не GPU transport.
- После всех тестов нет `NVRM Xid`, IOMMU faults, PCIe AER/uncorrected/fatal errors или failed systemd units. GPU idle, пользовательских GPU-процессов нет; `nvidia-tune` вернул штатный steady-state power limit 220 W на каждой карте.
- APT holds сохранились. `dkms status` ожидаемо отмечает отличие installed modules от stock DKMS build: это следствие ручной установки P2P-набора, а не поломка. До локального DKMS/deb нельзя снимать holds или обновлять kernel/driver без повторной сборки и проверки.

Логи post-reboot тестов сохранены в `/home/alexey/driver-backups/nvidia-p2p-610-20260710T111607Z/`, а llama smoke — в подкаталоге `baselines/` этого backup-набора.

## Источники и замечания по риску

- Fork и инструкция: <https://github.com/aikitoria/open-gpu-kernel-modules>
- P2P на 3090 и значение полного 32G BAR1: <https://github.com/aikitoria/open-gpu-kernel-modules/issues/3>, <https://github.com/aikitoria/open-gpu-kernel-modules/issues/22>
- Пример ложного `topo: OK` при ошибке вычислений: <https://github.com/aikitoria/open-gpu-kernel-modules/issues/25>
- NVIDIA CUDA 13.3 release notes: <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/>
- NVIDIA CUDA compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>
- Официальные open kernel modules 610.43.02: <https://github.com/NVIDIA/open-gpu-kernel-modules>
