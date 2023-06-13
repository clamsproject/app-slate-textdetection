
# app-slate-textdetection

## User instruction

General user instruction for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp/).

We provide a Containerfile. If you want to run this tool as a docker container (not worrying about dependencies), build an image from the `Containerfile` using the following command:

```bash
docker build . -f Containerfile -t slate-text_detection
```

You can then run the image, with the target directory mounted to `/data`. MAKE SURE that the target directory is writable (`chmod o+w TARGET_DIR`). Port 5000 is the default:

```bash
docker run -v /path/to/files:/data -p 5000:5000 slate-text_detection
```

From here, it is simple to post to the docker image with the necessary Mmif files:

```bash
curl -X POST -d @path/to/mmif//file http://0.0.0.0:5000
```

### System requirments

This app does not require any additional software dependencies, besides those listed in the `requirements.txt`, which are explained within the CLAMS Apps Documentation.

### Configurable runtime parameters

This app does not contain any specific runtime parameters for user functionality.
