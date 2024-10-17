# Define the URLs for the checkpoints
$baseUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
$sam2_hiera_t_url = "${baseUrl}sam2.1_hiera_tiny.pt"
$sam2_hiera_s_url = "${baseUrl}sam2.1_hiera_small.pt"
$sam2_hiera_b_plus_url = "${baseUrl}sam2.1_hiera_base_plus.pt"
$sam2_hiera_l_url = "${baseUrl}sam2.1_hiera_large.pt"

# Download each of the four checkpoints using Invoke-WebRequest
Write-Output "Downloading sam2_hiera_tiny.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_t_url -OutFile "checkpoints\2.1\sam2_hiera_tiny.pt" -ErrorAction Stop

Write-Output "Downloading sam2_hiera_small.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_s_url -OutFile "checkpoints\2.1\sam2_hiera_small.pt" -ErrorAction Stop

Write-Output "Downloading sam2_hiera_base_plus.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_b_plus_url -OutFile "checkpoints\2.1\sam2_hiera_base_plus.pt" -ErrorAction Stop

Write-Output "Downloading sam2_hiera_large.pt checkpoint..."
Invoke-WebRequest -Uri $sam2_hiera_l_url -OutFile "checkpoints\2.1\sam2_hiera_large.pt" -ErrorAction Stop

Write-Output "All checkpoints are downloaded successfully."


Write-Output "Downloading config files..."

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_t.yaml" -OutFile "sam2\sam2.1_hiera_t.yaml"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_s.yaml" -OutFile "sam2\sam2.1_hiera_s.yaml"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" -OutFile "sam2\sam2.1_hiera_l.yaml"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/sam2/05d9e57fb3945b10c861046c1e6749e2bfc258e3/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml" -OutFile "sam2\sam2.1_hiera_b+.yaml"

Write-Output "Config files downloaded successfully."


