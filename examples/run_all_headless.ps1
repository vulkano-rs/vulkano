foreach ($example in Get-ChildItem -Directory -Name) {
    $proc = Start-Process "cargo" "run --bin $example" -NoNewWindow -PassThru

    $timeouted = $null

    $proc | Wait-Process -Timeout 15 -ErrorAction SilentlyContinue -ErrorVariable timeouted

    if ($timeouted) {
        $proc | Stop-Process
    } elseif ($proc.ExitCode -ne 0) {
        Exit $proc.ExitCode
    }
}
