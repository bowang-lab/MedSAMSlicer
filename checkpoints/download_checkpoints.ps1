# Define the URLs for the checkpoints
$baseUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
$sam2_hiera_t_url = "${baseUrl}sam2_hiera_tiny.pt"
$sam2_hiera_s_url = "${baseUrl}sam2_hiera_small.pt"
$sam2_hiera_b_plus_url = "${baseUrl}sam2_hiera_base_plus.pt"
$sam2_hiera_l_url = "${baseUrl}sam2_hiera_large.pt"

# Download each of the four checkpoints using Invoke-WebRequest
Write-Output "Downloading sam2_hiera_tiny.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_t_url -OutFile "sam2_hiera_tiny.pt" -ErrorAction Stop

Write-Output "Downloading sam2_hiera_small.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_s_url -OutFile "sam2_hiera_small.pt" -ErrorAction Stop

Write-Output "Downloading sam2_hiera_base_plus.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_b_plus_url -OutFile "sam2_hiera_base_plus.pt" -ErrorAction Stop

Write-Output "Downloading sam2_hiera_large.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_l_url -OutFile "sam2_hiera_large.pt" -ErrorAction Stop

Write-Output "All checkpoints are downloaded successfully."