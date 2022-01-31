BatchNorm：batch方向做归一化，算N*H*W的均值


LayerNorm：channel方向做归一化，算C*H*W的均值


InstanceNorm：一个channel内做归一化，算H*W的均值


GroupNorm：将channel方向分group，然后每个group内做归一化，算(C//G)*H*W的均值 