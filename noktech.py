#!/usr/bin/env python
import os
import sys
import argparse
from scripts.trainer import ModelTrainer

def setup_directories() -> None:
    """Configura diretórios necessários"""
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/cache",
        "models"
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

def train(args: argparse.Namespace) -> None:
    """Executa treinamento dos modelos"""
    trainer = ModelTrainer(config_path=args.config)
    trainer.train()
    if args.export:
        trainer.export_to_gguf()

def main() -> None:
    """Função principal"""
    parser = argparse.ArgumentParser(description="NOK - The Liar Crow")
    
    # Argumentos globais
    parser.add_argument("--config", type=str, default="config.json",
                       help="Caminho para arquivo de configuração")
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Comando train
    train_parser = subparsers.add_parser("train", help="Treina os modelos")
    train_parser.add_argument("--export", action="store_true",
                            help="Exporta modelos para GGUF após treinamento")
    
    args = parser.parse_args()
    
    # Configura diretórios
    setup_directories()
    
    # Executa comando
    if args.command == "train":
        train(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
