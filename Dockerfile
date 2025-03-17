FROM tensorflow/serving:latest

# Copy model & konfigurasi monitoring
COPY ./output/serving_model /models/employee-performance-model
COPY ./config /model_config

# Set environment variables
ENV MODEL_NAME=employee-performance-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

# Expose ports for monitoring & API access
EXPOSE 8500 8501

# Create entrypoint script
RUN echo '#!/bin/bash \n\n\
set -e \n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
