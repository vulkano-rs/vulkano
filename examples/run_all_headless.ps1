foreach ($example in Get-ChildItem -Directory -Name) {
    $proc = Start-Process -FilePath "cargo" -ArgumentList "run --bin $example" -NoNewWindow -PassThru

    $timeouted = $null

    Wait-Process -InputObject $proc -Timeout 15 -ErrorAction SilentlyContinue -ErrorVariable timeouted

    if ($timeouted) {
        kill $proc
    } elseif ($proc.ExitCode -ne 0) {
        Exit $proc.ExitCode
    }
}
