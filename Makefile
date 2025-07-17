.ONESHELL:
SHELL=/bin/bash
PYTHON_INTERPRETER = python3
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


setup_env:
	conda create -y --name clip_fairness python=3.10
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	
requirements:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

get_all_embeddings: requirements
	$(CONDA_ACTIVATE) clip_fairness
	@echo GENERATING EMBEDDINGS WITH MEDCLIP
	$(PYTHON_INTERPRETER) apply_model.py --model_name medclip --batch_size 16

	@echo GENERATING EMBEDDINGS WITH MEDIMAGEINSIGHT
	$(PYTHON_INTERPRETER) apply_model.py --model_name medimageinsight --batch_size 16

	@echo GENERATING EMBEDDINGS WITH BIOVIL
	$(PYTHON_INTERPRETER) apply_model.py --model_name biovil --batch_size 16

	@echo GENERATING EMBEDDINGS WITH BIOVIL-T
	$(PYTHON_INTERPRETER) apply_model.py --model_name biovil-t --batch_size 16

	@echo GENERATING EMBEDDINGS WITH CHEXZERO
	$(PYTHON_INTERPRETER) apply_model.py --model_name chexzero --batch_size 16

	@echo GENERATING EMBEDDINGS WITH CXR-CLIP
	$(PYTHON_INTERPRETER) apply_model.py --model_name cxrclip --batch_size 16

get_embeddings:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) apply_model.py --model_name $(model_name) --batch_size 16

create_pca:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name medclip --projection_type PCA
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name medimageinsight --projection_type PCA
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name biovil --projection_type PCA
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name biovil-t --projection_type PCA
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name chexzero --projection_type PCA
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name cxrclip --projection_type PCA

create_tsne:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name medclip --projection_type TSNE
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name medimageinsight --projection_type TSNE
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name biovil --projection_type TSNE
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name biovil-t --projection_type TSNE
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name chexzero --projection_type TSNE
	$(PYTHON_INTERPRETER) embedding_analysis.py --model_name cxrclip --projection_type TSNE

compute_probas_and_evaluate_performance:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name medclip --compute_probas True --batch_size 16
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name medimageinsight --compute_probas True --batch_size 16
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name biovil --compute_probas True --batch_size 16
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name biovil-t --compute_probas True --batch_size 16
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name chexzero --compute_probas True --batch_size 16
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name cxrclip --compute_probas True --batch_size 16

evaluate_performance:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name medclip
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name medimageinsight
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name biovil
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name biovil-t
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name chexzero
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name cxrclip

evaluate_performance_drains:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name medclip --dataset CXR14
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name medimageinsight --dataset CXR14
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name biovil --dataset CXR14
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name biovil-t --dataset CXR14
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name chexzero --dataset CXR14
	$(PYTHON_INTERPRETER) evaluate_performance.py --model_name cxrclip --dataset CXR14


generate_figures_tables:
	$(CONDA_ACTIVATE) clip_fairness
	$(PYTHON_INTERPRETER) generate_figures_tables.py

predict_attribute:
	$(CONDA_ACTIVATE) clip_fairness
	python predict_attribute.py --modality image --attribute sex_label --model_name medclip --classification_head mlp
	python predict_attribute.py --modality image --attribute race_label --model_name medclip --classification_head mlp
	python predict_attribute.py --modality image --attribute age_label --model_name medclip --classification_head mlp

	python predict_attribute.py --modality image --attribute sex_label --model_name biovil --classification_head mlp
	python predict_attribute.py --modality image --attribute race_label --model_name biovil --classification_head mlp
	python predict_attribute.py --modality image --attribute age_label --model_name biovil --classification_head mlp

	python predict_attribute.py --modality image --attribute sex_label --model_name biovil-t --classification_head mlp
	python predict_attribute.py --modality image --attribute race_label --model_name biovil-t --classification_head mlp
	python predict_attribute.py --modality image --attribute age_label --model_name biovil-t --classification_head mlp

	python predict_attribute.py --modality image --attribute sex_label --model_name medimageinsight --classification_head mlp
	python predict_attribute.py --modality image --attribute race_label --model_name medimageinsight --classification_head mlp
	python predict_attribute.py --modality image --attribute age_label --model_name medimageinsight --classification_head mlp

	python predict_attribute.py --modality image --attribute sex_label --model_name chexzero --classification_head mlp
	python predict_attribute.py --modality image --attribute race_label --model_name chexzero --classification_head mlp
	python predict_attribute.py --modality image --attribute age_label --model_name chexzero --classification_head mlp

	python predict_attribute.py --modality image --attribute sex_label --model_name cxrclip --classification_head mlp
	python predict_attribute.py --modality image --attribute race_label --model_name cxrclip --classification_head mlp
	python predict_attribute.py --modality image --attribute age_label --model_name cxrclip --classification_head mlp