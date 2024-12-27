from typing import Dict, Any, List, Optional, Tuple, Union
import os
from pathlib import Path
from datetime import datetime, timezone
import torch
from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from peft.tuners.lora import LoraConfig
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from huggingface_hub import login
from .database import TrainingSession, ModelSession, TrainingMetrics, get_db
from .config import load_config, AppConfig, TrainingModelConfig

# Carrega variáveis de ambiente
load_dotenv()

class DataProcessor:
    """Classe responsável pelo processamento de dados"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Configura os diretórios necessários"""
        os.makedirs(self.config.data.directories.raw, exist_ok=True)
        os.makedirs(self.config.data.directories.processed, exist_ok=True)
        os.makedirs(self.config.data.directories.cache, exist_ok=True)
        os.makedirs(self.config.data.directories.models, exist_ok=True)
        os.makedirs(self.config.data.directories.exports, exist_ok=True)
            
    def _get_data_files(self, pattern: str) -> List[str]:
        """Obtém lista de arquivos baseado no padrão"""
        raw_dir = Path(self.config.data.directories.raw)
        files = [str(f) for f in raw_dir.glob(pattern)]
        return sorted(files) if not self.config.data.processing.shuffle_files else files
        
    def prepare_datasets(self, tokenizer: PreTrainedTokenizer, model_config: TrainingModelConfig) -> Dict[str, HFDataset]:
        """Prepara datasets para treinamento"""
        datasets: Dict[str, HFDataset] = {}
        
        # Função de tokenização
        def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=model_config.max_sequence_length,
                return_tensors="pt"
            )
            
        # Processa cada tipo de dataset
        for split, pattern in [("train", "*.jsonl")]:
            files = self._get_data_files(pattern)
            if not files:
                continue
                
            # Define caminho para cache
            cache_dir = Path(self.config.data.directories.processed) / split / model_config.name.replace("/", "_")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Verifica se existe versão processada
            processed_file = cache_dir / "dataset.arrow"
            if processed_file.exists() and not self.config.data.processing.overwrite_cache:
                print(f"Carregando dataset processado de {processed_file}")
                dataset = load_dataset(
                    "arrow",
                    data_files=str(processed_file),
                    cache_dir=self.config.data.directories.cache
                )
                if isinstance(dataset, DatasetDict):
                    dataset = dataset["train"]
            else:
                # Carrega dataset
                dataset = load_dataset(
                    "json",
                    data_files=files,
                    num_proc=self.config.data.processing.num_workers,
                    cache_dir=self.config.data.directories.cache
                )
                if isinstance(dataset, DatasetDict):
                    dataset = dataset["train"]
                
                # Aplica tokenização
                dataset = dataset.map(  # type: ignore
                    tokenize_function,
                    batched=True,
                    num_proc=self.config.data.processing.num_workers,
                    remove_columns=dataset.column_names,  # type: ignore
                    cache_file_name=str(cache_dir / "tokenized.arrow")
                )
                
                # Salva dataset processado
                dataset.save_to_disk(str(processed_file))  # type: ignore
                print(f"Dataset processado salvo em {processed_file}")
            
            datasets[split] = dataset
            
        return datasets

class ModelTrainer:
    """Classe responsável pelo treinamento do modelo"""
    
    def __init__(self, config_path: str = "config.json"):
        # Autentica no Hugging Face
        hf_token = os.getenv("huggingface_token")
        if not hf_token:
            raise ValueError("Token do Hugging Face não encontrado no arquivo .env")
        login(token=hf_token)
        
        self.config = load_config(config_path)
        self.device = self._setup_device()
        self.db: Session = get_db()
        self.training_session: Optional[TrainingSession] = None
        
        # Inicializa processador de dados
        self.data_processor = DataProcessor(self.config)
        
    def _setup_device(self) -> str:
        """Configura dispositivo de treinamento"""
        if self.config.hardware.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.hardware.device
        
    def _setup_model_and_tokenizer(self, model_config: TrainingModelConfig) -> Tuple[Union[AutoModelForCausalLM, PeftModel, PeftMixedModel], PreTrainedTokenizer]:
        """Configura modelo e tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.name,
            trust_remote_code=model_config.optimization.trust_remote_code
        )
        
        # Carrega modelo base
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name,
            torch_dtype=torch.float16 if model_config.optimization.use_fp16 and self.device == "cuda" else torch.float32,
            load_in_4bit=model_config.optimization.use_4bit,
            load_in_8bit=model_config.optimization.use_8bit,
            trust_remote_code=model_config.optimization.trust_remote_code,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Aplica otimizações
        if model_config.optimization.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        # Configura LoRA
        lora_config = LoraConfig(
            r=model_config.lora.r,
            lora_alpha=model_config.lora.alpha,
            target_modules=model_config.lora.target_modules,
            lora_dropout=model_config.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
            
        return model, tokenizer
            
    def _create_training_session(self) -> TrainingSession:
        """Cria nova sessão de treinamento"""
        session = TrainingSession(
            start_time=datetime.now(timezone.utc),
            config=self.config.model_dump(),
            status="running"
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        self.training_session = session
        return session
        
    def _create_model_session(self, model_config: TrainingModelConfig) -> ModelSession:
        """Cria nova sessão para um modelo específico"""
        if not self.training_session:
            raise ValueError("Sessão de treinamento não iniciada")
            
        session = ModelSession(
            training_session_id=self.training_session.id,
            model_name=model_config.name,
            start_time=datetime.now(timezone.utc),
            config=model_config.model_dump(),
            status="running"
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
        
    def _log_metrics(self, model_session: ModelSession, epoch: float, step: int, loss: float, learning_rate: float) -> None:
        """Registra métricas de treinamento"""
        metrics = TrainingMetrics(
            model_session_id=model_session.id,
            epoch=int(epoch),
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            timestamp=datetime.now(timezone.utc)
        )
        self.db.add(metrics)
        self.db.commit()
        
    def train(self) -> None:
        """Executa treinamento dos modelos"""
        try:
            # Inicia sessão de treinamento
            self._create_training_session()
            
            # Treina cada modelo
            for model_config in self.config.training:
                try:
                    # Cria sessão para o modelo
                    model_session = self._create_model_session(model_config)
                    
                    # Configura modelo e tokenizer
                    model, tokenizer = self._setup_model_and_tokenizer(model_config)
                    
                    # Prepara datasets
                    datasets = self.data_processor.prepare_datasets(tokenizer, model_config)
                    if not datasets.get("train"):
                        raise ValueError(f"Nenhum dado de treinamento encontrado para {model_config.name}")
                    
                    # Configura argumentos de treinamento
                    training_args = TrainingArguments(
                        output_dir=os.path.join(self.config.model.training.output_dir, model_config.name.replace("/", "_")),
                        num_train_epochs=self.config.model.training.num_train_epochs,
                        per_device_train_batch_size=self.config.model.training.per_device_train_batch_size,
                        learning_rate=self.config.model.training.learning_rate,
                        weight_decay=self.config.model.training.weight_decay,
                        warmup_steps=self.config.model.training.warmup_steps,
                        logging_steps=self.config.model.training.logging_steps,
                        save_steps=self.config.model.training.save_steps,
                        gradient_accumulation_steps=self.config.model.training.gradient_accumulation_steps,
                        max_grad_norm=self.config.model.training.max_grad_norm,
                        fp16=model_config.optimization.use_fp16 and self.device == "cuda",
                        report_to=self.config.logging.report_to,
                        save_total_limit=self.config.logging.save_total_limit
                    )
                    
                    # Configura data collator
                    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer,
                        mlm=False
                    )
                    
                    # Configura trainer com callback para métricas
                    class MetricsCallback(Trainer):
                        def __init__(self, parent: 'ModelTrainer', model_session: ModelSession, **kwargs: Any):
                            super().__init__(**kwargs)
                            self.parent = parent
                            self.model_session = model_session
                            
                        def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: Optional[Dict[str, float]] = None, **kwargs: Any) -> None:
                            if state.is_local_process_zero and logs:
                                self.parent._log_metrics(
                                    self.model_session,
                                    state.epoch or 0,
                                    state.global_step or 0,
                                    logs.get("loss", 0.0),
                                    state.learning_rate or 0.0
                                )
                    
                    # Configura e executa trainer
                    trainer = MetricsCallback(
                        parent=self,
                        model_session=model_session,
                        model=model,
                        args=training_args,
                        train_dataset=datasets["train"],
                        data_collator=data_collator
                    )
                    trainer.train()
                    
                    # Atualiza status da sessão do modelo
                    model_session.status = "completed"  # type: ignore
                    model_session.end_time = datetime.now(timezone.utc)  # type: ignore
                    self.db.commit()
                    
                except Exception as e:
                    # Atualiza status da sessão do modelo em caso de erro
                    model_session.status = "failed"  # type: ignore
                    model_session.end_time = datetime.now(timezone.utc)  # type: ignore
                    self.db.commit()
                    print(f"Erro no treinamento do modelo {model_config.name}: {str(e)}")
                    
            # Atualiza status da sessão de treinamento
            if self.training_session:
                self.training_session.status = "completed"  # type: ignore
                self.training_session.end_time = datetime.now(timezone.utc)  # type: ignore
                self.db.commit()
                
        except Exception as e:
            if self.training_session:
                self.training_session.status = "failed"  # type: ignore
                self.training_session.end_time = datetime.now(timezone.utc)  # type: ignore
                self.db.commit()
            raise e
            
    def export_to_gguf(self) -> None:
        """Exporta modelos para formato GGUF"""
        # TODO: Implementar exportação para GGUF
        pass

def main() -> None:
    trainer = ModelTrainer()
    trainer.train()
    trainer.export_to_gguf()

if __name__ == "__main__":
    main() 