import json
import os
import typer
from typing import Optional
from pathlib import Path
import logging
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.llm_inference import call_ollama
from src.prompt_builder import build_prompt
from src.retriever import Retriever
from src.embed_index import EmbedIndexer
from src.preprocessing import DataPreprocessor

os.environ["TOKENIZERS_PARALLELISM"] = "true"

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

app = typer.Typer(
    name="faq-toolkit",
    help="A toolkit for processing and validating FAQ data",
    add_completion=False
)

@app.command()
def preprocess(
    input_file: Path = typer.Argument(
        ..., 
        exists=True, 
        dir_okay=False, 
        readable=True,
        help="Input JSON file with FAQ data"
    ),
    output_file: Path = typer.Argument(
        ...,
        dir_okay=False,
        help="Output JSON file for cleaned data"
    ),
    report_invalid: bool = typer.Option(
        True,
        "--report-invalid/--no-report-invalid",
        help="Report invalid items instead of raising an error"
    ),
    report_file: Optional[Path] = typer.Option(
        None,
        "--report-file",
        "-r",
        help="File to save validation report to"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    )
):
    """
    Preprocess FAQ data:
    - Load JSON data and validate against schema
    - Clean text (remove HTML, trim whitespace, etc.)
    - Remove duplicate entries
    - Save cleaned data to output file
    """
    # Set logging level based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            # Create preprocessing task
            task = progress.add_task("[green]Preprocessing FAQ data...", total=None)
            
            # Initialize preprocessor
            preprocessor = DataPreprocessor(
                str(input_file),
                str(output_file),
                report_invalid=report_invalid,
                report_filepath=str(report_file) if report_file else None
            )
            
            # Run preprocessing
            df = preprocessor.run()
            
            # Complete the task
            progress.update(task, completed=True)
        
        # Show summary
        console.print(f"\n[bold green]✓[/] Successfully processed FAQ data:")
        console.print(f"  • Input file: [bold]{input_file}[/]")
        console.print(f"  • Output file: [bold]{output_file}[/]")
        console.print(f"  • Items processed: [bold]{len(df)}[/]")
        
        if report_invalid and report_file:
            console.print(f"  • Validation report: [bold]{report_file}[/]")
            
    except Exception as e:
        console.print(f"\n[bold red]✗[/] Error: {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)

@app.command()
def validate(
    input_file: Path = typer.Argument(
        ..., 
        exists=True, 
        dir_okay=False, 
        readable=True,
        help="Input JSON file with FAQ data to validate"
    ),
    report_file: Optional[Path] = typer.Option(
        None,
        "--report-file",
        "-r",
        help="File to save validation report to"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    )
):
    """
    Validate FAQ data against schema without preprocessing.
    Useful for checking if data meets requirements before processing.
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[green]Validating FAQ data...", total=None)
            
            dummy_output = "dummy_output.json"
            preprocessor = DataPreprocessor(
                str(input_file),
                dummy_output,
                report_invalid=True,
                report_filepath=str(report_file) if report_file else None
            )
            
            raw_data = preprocessor.load_data()
            result = preprocessor.validate_data(raw_data)
            
            progress.update(task, completed=True)
        
        valid_count = len(result)
        total_count = len(raw_data)
        invalid_count = total_count - valid_count
        
        if invalid_count == 0:
            console.print(f"\n[bold green]✓[/] All {total_count} items are valid!")
        else:
            console.print(f"\n[bold yellow]![/] Found {invalid_count} invalid items out of {total_count} total items.")
            
        if report_file:
            console.print(f"  • Validation report saved to: [bold]{report_file}[/]")
            
    except Exception as e:
        console.print(f"\n[bold red]✗[/] Error: {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command("build-index")
def build_index(
    input_file: Path = typer.Argument(
        ..., 
        exists=True, 
        dir_okay=False, 
        readable=True,
        help="Input JSON file with cleaned FAQ data"
    ),
    index_file: Path = typer.Argument(
        ...,
        dir_okay=False,
        help="Output file to save the FAISS index"
    ),
    model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--model",
        "-m",
        help="Embedding model to use (default: all-MiniLM-L6-v2)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    )
):
    """
    Build an embedding index from cleaned FAQ data.
    
    This command:
      - Loads the cleaned JSON file.
      - Extracts the 'question' field from each FAQ.
      - Generates embeddings using the specified SentenceTransformer model.
      - Builds a FAISS index with cosine similarity (via normalized inner product).
      - Saves the FAISS index to disk.
    """
    if verbose:
        console.log("Verbose mode enabled.")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[green]Building embedding index...", total=None)
            
            # Load cleaned FAQ data
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract questions from the FAQ items
            questions = [item["question"] for item in data if "question" in item]
            if not questions:
                console.print("[red]No valid questions found in the input file.[/red]")
                raise typer.Exit(1)
            
            # Instantiate the embedding/indexer class
            embed_indexer = EmbedIndexer(model_name=model, index_path=str(index_file))
            
            # Generate embeddings for the questions
            embeddings = embed_indexer.embed_texts(questions)
            
            # Build a FAISS index using the embeddings
            index = embed_indexer.build_index(embeddings)
            
            # Save the index to the given file path
            embed_indexer.save_index(index, str(index_file))
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✓ Successfully built and saved FAISS index to {index_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error building index: {e}[/red]")
        raise typer.Exit(1)

@app.command("retrieve")
def retrieve(
    query: str = typer.Argument(..., help="User query to retrieve relevant FAQ items."),
    index_file: Path = typer.Option(..., help="Path to the saved FAISS index file."),
    data_file: Path = typer.Option(..., help="Path to the cleaned FAQ JSON file."),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Embedding model to use."),
    top_k: int = typer.Option(3, "--top_k", "-k", help="Number of results to retrieve."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
):
    """
    Retrieve top FAQ items relevant to the user query.
    """
    try:
        retriever = Retriever(index_path=str(index_file), data_path=str(data_file), model=model)
        results = retriever.retrieve(query, top_k)
        if not results:
            logging.warning("No results found.")
            raise typer.Exit(1)
        for i, res in enumerate(results, start=1):
            logging.info(f"{i}. Score: {res['score']:.4f}")
            logging.info(f"   Q: {res['question']}")
            logging.info(f"   A: {res['answer']}\n")
    except Exception as e:
        console.print(f"[red]Error during retrieval: {e}[/red]")
        raise typer.Exit(1)

@app.command("llm-infer")
def llm_infer(
    query: str = typer.Argument(..., help="User query for LLM inference."),
    index_file: Path = typer.Option(..., help="Path to the saved FAISS index file."),
    data_file: Path = typer.Option(..., help="Path to the cleaned FAQ JSON file."),
    model: str = typer.Option("all-MiniLM-L6-v2", "--embed-model", "-m", help="Embedding model to use."),
    llm_model: str = typer.Option("Mistral", "--llm-model", help="LLM model name for Ollama."),
    top_k: int = typer.Option(3, "--top_k", "-k", help="Number of retrieval results to use for context."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
):
    """
    Retrieve context using vector search, build a prompt, and perform LLM inference.
    """
    try:
        # Step 1: Retrieve context
        retriever = Retriever(index_path=str(index_file), data_path=str(data_file), model=model)
        retrieved = retriever.retrieve(query, top_k)
        if not retrieved:
            logging.warning("No context retrieved. Exiting.")
            raise typer.Exit(1)
        
        # Step 2: Build RAG-style prompt
        prompt = build_prompt(retrieved, query)
        if verbose:
            console.print("[blue]Constructed prompt:[/blue]")
            console.print(prompt)
        
        # Step 3: Call LLM inference via Ollama
        answer = call_ollama(prompt, model=llm_model)
        logging.info("LLM Answer:\n%s", answer)
        
    except Exception as e:
        console.print(f"[red]Error during LLM inference: {e}[/red]")
        raise typer.Exit(1)

@app.command("prepare-model")
def prepare_model(
    input_file: Path = typer.Argument(
        ..., 
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input JSON file with cleaned FAQ data"
    ),
    index_file: Path = typer.Argument(
        ...,
        dir_okay=False,
        help="Output file to save the FAISS index"
    ),
    embed_model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--embed-model",
        "-e",
        help="Embedding model to use for training (default: all-MiniLM-L6-v2)"
    ),
    llm_model: str = typer.Option(
        "Mistral",
        "--llm-model",
        "-l",
        help="LLM model name for Ollama (default: Mistral)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    )
):
    """
    Prepare and train the retrieval model:
      - Build the FAISS index from cleaned FAQ data.
      - Perform a readiness check of the LLM interface (Ollama).
    
    This command will exit with success once the model is prepared and the LLM is ready.
    """
    try:
        # Build index from cleaned FAQ data.
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[green]Building embedding index...", total=None)
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            questions = [item["question"] for item in data if "question" in item]
            if not questions:
                logger.error("No valid questions found in the input file.")
                raise typer.Exit(1)
            embed_indexer = EmbedIndexer(model_name=embed_model, index_path=str(index_file))
            embeddings = embed_indexer.embed_texts(questions)
            index = embed_indexer.build_index(embeddings)
            embed_indexer.save_index(index, str(index_file))
            progress.update(task, completed=True)
        
        logging.info("Successfully built and saved FAISS index to %s", index_file)
        
        # LLM readiness check: Use a simple dummy prompt.
        dummy_prompt = "ping"
        logging.info("Performing LLM readiness check using prompt: '%s'", dummy_prompt)
        try:
            response = call_ollama(dummy_prompt, model=llm_model)
            logging.info("LLM readiness check successful. Response: %s", response)
        except Exception as e:
            logging.error("LLM readiness check failed: %s", e)
            raise typer.Exit(1)
        
        logging.info("[bold green]Model preparation complete.[/]")
        
    except Exception as e:
        logging.error("Error during model preparation: %s", e)
        raise typer.Exit(1)


@app.command("get-result")
def get_result(
    query: str = typer.Argument(..., help="User query for which to obtain an answer."),
    index_file: Path = typer.Option(..., help="Path to the saved FAISS index file."),
    data_file: Path = typer.Option(..., help="Path to the cleaned FAQ JSON file."),
    embed_model: str = typer.Option("all-MiniLM-L6-v2", "--embed-model", "-e", help="Embedding model to use."),
    llm_model: str = typer.Option("Mistral", "--llm-model", "-l", help="LLM model name for Ollama."),
    top_k: int = typer.Option(3, "--top_k", "-k", help="Number of retrieval results to use for context."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
):
    """
    Retrieve context using vector search, build a prompt, and perform LLM inference.
    
    This command assumes the model is already prepared.
    """
    try:
        # Retrieve context.
        retriever = Retriever(index_path=str(index_file), data_path=str(data_file), model=embed_model)
        retrieved = retriever.retrieve(query, top_k)
        if not retrieved:
            logging.error("No context retrieved. Exiting.")
            raise typer.Exit(1)
        
        # Build RAG-style prompt.
        prompt = build_prompt(retrieved, query)
        if verbose:
            logging.info("Constructed prompt:\n%s", prompt)
        
        # Call LLM inference.
        answer = call_ollama(prompt, model=llm_model)
        logging.info("LLM Answer:\n%s", answer)
    except Exception as e:
        logging.error("Error during LLM inference: %s", e)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()