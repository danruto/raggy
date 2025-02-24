{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    ml-pkgs.url = "github:nixvital/ml-pkgs";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    { nixpkgs
    , flake-utils
    , self
    , ...
    }@inputs: {
      overlays.dev = nixpkgs.lib.composeManyExtensions [
        inputs.ml-pkgs.overlays.torch-family
      ];
    } // flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [ self.overlays.dev ];
      };
    in
    {
      devShells.default =
        let
          python-env = pkgs.python3.withPackages (pyPkgs: with pyPkgs;
            [
              numpy
              pandas
              # pytorchWithCuda11
              # pytorchWithoutCuda
              # transformers
              # xformers
              # scipy
              # emoji
              # torch

              # fastapi
              # uvicorn

              # Uncomment things below if you need them

              # torchvisionWithCuda11
              # pytorchLightningWithCuda11

              # Random useful items
              dateutil
              # gql
              # requests-toolbelt
              # openai
              # python-dotenv
              # faker

              langchain
              langchain-core
              langchain-community
              langchain-ollama
              langchain-text-splitters
              streamlit
              fastembed
              pypdf
              chromadb
            ]);
        in
        pkgs.mkShell {

          packages = with pkgs; [
            fixjson
            dockerfile-language-server-nodejs
            pyright
            black
            ruff
            ruff-lsp

            python-env
            # poetry
            just
          ];
        };
    });
}
