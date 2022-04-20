# Stochastic Procedural Texture Shader for Godot Engine

This is implement of the Stochastic Procedural Texture Shader for Godot Engine.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/E1E44AWTA)

## How to use

### precomp_tex.py

* Python 3.9.10以降と numpy, opencv が必要です。

以下のコマンドで、事前計算済みのテクスチャとテーブルを生成します。

```python precomp_tex.py -i [textureへのパス]```

-hでヘルプが見られるので、詳しくはそちらを参照してください。

### シェーダー

入力する事前計算済みのテクスチャの設定は以下のとおりにして再インポートしてください。

* *_t_input.png
    * Compress/Mode: Lossless もしくは Lossy
    * Flags/Repeat: Enabled
    * Flags/Filter: True
    * Flags/Mipmaps: True（必要なら）
* *_lut.png
    * Compress/Mode: Lossless
    * Flags/Repeat: Disabled
    * Flags/Filter: False
    * Flags/Mipmaps: False

## TODO

* [See issues](https://bitbucket.org/arlez80/stochastic-procedural-texture-shader/issues?status=new&status=open)

## References

* [High-Performance By-Example Noise using a Histogram-Preserving Blending Operator](https://eheitzresearch.wordpress.com/722-2/)
* [Procedural Stochastic Textures by Tiling and Blending](https://eheitzresearch.wordpress.com/738-2/)

## License

MIT License

## Author

あるる / きのもと 結衣 @arlez80
