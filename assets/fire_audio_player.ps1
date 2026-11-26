param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$MediaUri,
    [Parameter(Mandatory = $true, Position = 1)]
    [string]$CommandPath
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName PresentationCore

$script:player = New-Object System.Windows.Media.MediaPlayer
$script:player.Volume = 1.0
$script:player.IsMuted = $false
$script:mediaReady = $false
$script:playRequested = $false
$script:isPlaying = $false
$script:commandPath = $CommandPath

$script:player.add_MediaOpened({
    $script:mediaReady = $true
    if ($script:playRequested) {
        $script:player.Position = [TimeSpan]::Zero
        $script:player.Play()
        $script:isPlaying = $true
        $script:playRequested = $false
    }
})
$script:player.add_MediaEnded({
    if ($script:isPlaying) {
        $script:player.Position = [TimeSpan]::Zero
        $script:player.Play()
    }
})

$script:timer = New-Object System.Windows.Threading.DispatcherTimer
$script:timer.Interval = [TimeSpan]::FromMilliseconds(50)
$script:timer.add_Tick({
    if (-not (Test-Path -LiteralPath $script:commandPath)) { return }
    try {
        $command = (Get-Content -LiteralPath $script:commandPath -Raw).Trim()
        Remove-Item -LiteralPath $script:commandPath -Force
    }
    catch { return }

    if ($command -eq "PLAY") {
        $script:playRequested = $true
        if ($script:mediaReady) {
            $script:player.Position = [TimeSpan]::Zero
            $script:player.Play()
            $script:isPlaying = $true
            $script:playRequested = $false
        }
    }
    elseif ($command -eq "STOP") {
        $script:playRequested = $false
        $script:isPlaying = $false
        $script:player.Stop()
    }
    elseif ($command -eq "EXIT") {
        $script:playRequested = $false
        $script:isPlaying = $false
        $script:player.Stop()
        $script:timer.Stop()
        [System.Windows.Threading.Dispatcher]::CurrentDispatcher.BeginInvokeShutdown(
            [System.Windows.Threading.DispatcherPriority]::Normal)
    }
})

$script:player.Open([Uri]$MediaUri)
$script:timer.Start()
[System.Windows.Threading.Dispatcher]::Run()
$script:player.Close()
