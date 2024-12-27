from typing import List, Optional, Union, Literal
from pathlib import Path
import json
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo

def default_training_models() -> List["TrainingModelConfig"]:
    return []

def default_modules() -> List[str]:
    return ["q_proj", "v_proj"]

def default_formats() -> List[str]:
    return ["jsonl", "txt"]

def default_loggers() -> List[str]:
    return ["tensorboard", "wandb"]

class ModelArchitecture(BaseModel):
    """Configuração da arquitetura do modelo"""
    base_model_type: Literal["mistral", "llama"] = Field(..., description="Tipo do modelo base")
    vocab_size: int = Field(32000, ge=1000, description="Tamanho do vocabulário")
    hidden_size: int = Field(4096, ge=128, description="Dimensão dos embeddings")
    intermediate_size: int = Field(11008, ge=512, description="Dimensão da camada intermediária")
    num_hidden_layers: int = Field(32, ge=1, description="Número de camadas")
    num_attention_heads: int = Field(32, ge=1, description="Número de cabeças de atenção")
    max_sequence_length: int = Field(2048, ge=128, description="Tamanho máximo da sequência")
    pad_token_id: int = Field(0, ge=0, description="ID do token de padding")
    bos_token_id: int = Field(1, ge=0, description="ID do token de início")
    eos_token_id: int = Field(2, ge=0, description="ID do token de fim")
    
    @field_validator("intermediate_size")
    @classmethod
    def validate_intermediate_size(cls, v: int, info: ValidationInfo) -> int:
        hidden_size = info.data.get("hidden_size", 0)
        if hidden_size and v < hidden_size:
            raise ValueError("intermediate_size deve ser maior que hidden_size")
        return v
    
    @field_validator("num_attention_heads")
    @classmethod
    def validate_attention_heads(cls, v: int, info: ValidationInfo) -> int:
        hidden_size = info.data.get("hidden_size", 0)
        if hidden_size and hidden_size % v != 0:
            raise ValueError("hidden_size deve ser divisível por num_attention_heads")
        return v

class ModelTrainingConfig(BaseModel):
    """Configuração de treinamento do modelo"""
    output_dir: str = Field("models", description="Diretório para salvar o modelo")
    num_train_epochs: int = Field(3, ge=1, description="Número de épocas")
    per_device_train_batch_size: int = Field(4, ge=1, description="Tamanho do batch")
    learning_rate: float = Field(2e-5, gt=0, description="Taxa de aprendizado")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay")
    warmup_steps: int = Field(500, ge=0, description="Passos de warmup")
    max_steps: Optional[int] = Field(None, ge=1, description="Máximo de passos")
    save_steps: int = Field(1000, ge=1, description="Frequência para salvar")
    logging_steps: int = Field(100, ge=1, description="Frequência de logging")
    gradient_accumulation_steps: int = Field(4, ge=1, description="Acumulação de gradiente")
    max_grad_norm: float = Field(1.0, gt=0, description="Clipping de gradiente")

class ModelOptimization(BaseModel):
    """Configurações de otimização"""
    use_gradient_checkpointing: bool = Field(True, description="Usar gradient checkpointing")
    use_flash_attention: bool = Field(True, description="Usar Flash Attention 2.0")
    use_fp16: bool = Field(True, description="Usar precisão FP16")
    use_4bit: bool = Field(False, description="Usar quantização 4-bit")
    use_8bit: bool = Field(False, description="Usar quantização 8-bit")
    trust_remote_code: bool = Field(True, description="Confiar em código remoto")
    
    @model_validator(mode='after')
    def validate_quantization(model: "ModelOptimization") -> "ModelOptimization":
        if model.use_4bit and model.use_8bit:
            raise ValueError("Não é possível usar quantização 4-bit e 8-bit simultaneamente")
        if model.use_4bit and model.use_fp16:
            raise ValueError("Não é possível usar quantização 4-bit com FP16")
        if model.use_8bit and model.use_fp16:
            raise ValueError("Não é possível usar quantização 8-bit com FP16")
        return model

class ModelExport(BaseModel):
    """Configurações de exportação"""
    format: Literal["gguf"] = Field("gguf", description="Formato de exportação")
    quantization: Literal["q4_k_m", "q4_0", "q5_k_m", "q5_0", "q8_0"] = Field("q4_k_m", description="Tipo de quantização")
    context_size: int = Field(2048, ge=128, description="Tamanho do contexto")

class ModelConfig(BaseModel):
    """Configuração do nosso modelo"""
    name: str = Field(..., pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", description="Nome do modelo")
    description: str = Field(..., min_length=10, description="Descrição do modelo")
    version: str = Field(..., pattern=r"^\d+\.\d+(\.\d+)?$", description="Versão do modelo")
    architecture: ModelArchitecture = Field(..., description="Configuração da arquitetura")
    training: ModelTrainingConfig = Field(..., description="Configuração de treinamento")
    optimization: ModelOptimization = Field(..., description="Configurações de otimização")
    export: ModelExport = Field(..., description="Configurações de exportação")

class LoraConfig(BaseModel):
    """Configuração do LoRA"""
    r: int = Field(..., ge=1, le=64, description="Rank do adaptador")
    alpha: int = Field(..., ge=1, description="Fator de escala")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Taxa de dropout")
    target_modules: List[str] = Field(default_factory=default_modules, description="Módulos alvo")
    
    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("target_modules não pode estar vazio")
        return v
    
    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: int, info: ValidationInfo) -> int:
        r = info.data.get("r", 0)
        if r and v < r:
            raise ValueError("alpha deve ser maior ou igual a r")
        return v

class TrainingModelConfig(BaseModel):
    """Configuração de um modelo de treinamento"""
    name: str = Field(..., pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", description="Nome do modelo no Hugging Face")
    description: str = Field(..., min_length=10, description="Descrição do modelo")
    max_sequence_length: int = Field(2048, ge=128, description="Tamanho máximo da sequência")
    optimization: ModelOptimization = Field(..., description="Configurações de otimização")
    lora: LoraConfig = Field(..., description="Configuração do LoRA")

class DataDirectories(BaseModel):
    """Configuração de diretórios"""
    raw: str = Field("data/raw", description="Diretório de dados brutos")
    processed: str = Field("data/processed", description="Diretório de dados processados")
    cache: str = Field("data/cache", description="Diretório de cache")
    models: str = Field("models", description="Diretório de modelos")
    exports: str = Field("exports", description="Diretório de exportação")

class DataProcessing(BaseModel):
    """Configuração de processamento"""
    shuffle_files: bool = Field(True, description="Embaralhar arquivos")
    combine_files: bool = Field(True, description="Combinar arquivos")
    overwrite_cache: bool = Field(False, description="Sobrescrever cache")
    num_workers: int = Field(4, ge=1, description="Número de workers")
    max_samples_per_file: Optional[int] = Field(None, ge=1, description="Limite de amostras")
    validation_split: float = Field(0.1, ge=0.0, le=0.5, description="Proporção de validação")
    test_split: float = Field(0.1, ge=0.0, le=0.5, description="Proporção de teste")
    
    @model_validator(mode='after')
    def validate_splits(model: "DataProcessing") -> "DataProcessing":
        total_split = model.validation_split + model.test_split
        if total_split >= 1.0:
            raise ValueError("A soma das proporções de validação e teste deve ser menor que 1.0")
        return model

class DataFormats(BaseModel):
    """Configuração de formatos"""
    input: List[str] = Field(default_factory=default_formats, description="Formatos de entrada")
    dataset: Literal["arrow", "parquet", "json"] = Field("arrow", description="Formato do dataset")
    
    @field_validator("input")
    @classmethod
    def validate_input(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("input não pode estar vazio")
        return v

class DataConfig(BaseModel):
    """Configuração de dados"""
    directories: DataDirectories = Field(..., description="Configuração de diretórios")
    processing: DataProcessing = Field(..., description="Configuração de processamento")
    formats: DataFormats = Field(..., description="Configuração de formatos")

class HardwareConfig(BaseModel):
    """Configuração de hardware"""
    device: Literal["auto", "cuda", "cpu", "mps"] = Field("auto", description="Dispositivo a usar")
    precision: Literal["auto", "fp32", "fp16", "bf16"] = Field("auto", description="Precisão a usar")
    gpu_memory_utilization: float = Field(0.9, ge=0.1, le=1.0, description="Utilização de memória GPU")
    cpu_offload: bool = Field(False, description="Usar offload para CPU")
    mixed_precision: bool = Field(True, description="Usar precisão mista")

class LoggingConfig(BaseModel):
    """Configuração de logging"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("INFO", description="Nível de log")
    save_strategy: Literal["steps", "epoch", "no"] = Field("steps", description="Estratégia de salvamento")
    save_steps: int = Field(1000, ge=1, description="Frequência de salvamento")
    save_total_limit: int = Field(5, ge=1, description="Limite de checkpoints")
    logging_steps: int = Field(100, ge=1, description="Frequência de logging")
    log_level: Literal["passive", "epoch", "strategy", "steps"] = Field("passive", description="Nível de log do trainer")
    report_to: List[str] = Field(default_factory=default_loggers, description="Ferramentas de logging")
    logging_dir: str = Field("logs", description="Diretório de logs")
    
    @field_validator("report_to")
    @classmethod
    def validate_report_to(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("report_to não pode estar vazio")
        return v

class AppConfig(BaseModel):
    """Configuração principal da aplicação"""
    model: ModelConfig = Field(..., description="Configuração do nosso modelo")
    training: List[TrainingModelConfig] = Field(default_factory=default_training_models, description="Lista de modelos para treinamento")
    data: DataConfig = Field(..., description="Configuração de dados")
    hardware: HardwareConfig = Field(..., description="Configuração de hardware")
    logging: LoggingConfig = Field(..., description="Configuração de logging")
    
    @field_validator("training")
    @classmethod
    def validate_training(cls, v: List[TrainingModelConfig]) -> List[TrainingModelConfig]:
        if not v:
            raise ValueError("training não pode estar vazio")
        return v
    
    @model_validator(mode='after')
    def validate_model_sequence(model: "AppConfig") -> "AppConfig":
        if not any("Mistral" in m.name for m in model.training):
            raise ValueError("Pelo menos um modelo de treinamento deve ser Mistral")
        return model

def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Carrega configuração do arquivo JSON"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return AppConfig.model_validate(data)

def save_config(config: AppConfig, config_path: Union[str, Path]) -> None:
    """Salva configuração em arquivo JSON"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.model_dump(), f, indent=2) 