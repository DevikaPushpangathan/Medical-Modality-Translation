def spectral_norm(module):
    return nn.utils.spectral_norm(module)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        self.down1 = self.conv_block(in_channels, features, batchnorm=False)
        self.down2 = self.conv_block(features, features*2)
        self.down3 = self.conv_block(features*2, features*4)
        self.down4 = self.conv_block(features*4, features*8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = self.up_conv(features*16, features*8)
        self.up3 = self.up_conv(features*8*2, features*4)
        self.up2 = self.up_conv(features*4*2, features*2)
        self.up1 = self.up_conv(features*2*2, features)
        self.final = nn.Conv2d(features*2, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_c, out_c, batchnorm=True):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def up_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        b = self.bottleneck(d4)
        u4 = self.up4(b); u4 = torch.cat([u4, d4], dim=1)
        u3 = self.up3(u4); u3 = torch.cat([u3, d3], dim=1)
        u2 = self.up2(u3); u2 = torch.cat([u2, d2], dim=1)
        u1 = self.up1(u2); u1 = torch.cat([u1, d1], dim=1)
        out = self.final(u1)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.tanh(out)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(features, features*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(features*2, features*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(features*4, 1, 4, 1, 1)),
        )

    def forward(self, x):
        return self.model(x)


class DeepDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(features, features*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(features*2, features*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(features*4, features*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(features*8, 1, 4, 1, 1))
        )

    def forward(self, x):
        return self.model(x)

G = UNetGenerator(in_channels=1, out_channels=1).to(device)
D1 = PatchDiscriminator(in_channels=2).to(device)
D2 = DeepDiscriminator(in_channels=2).to(device)

