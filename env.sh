#!/usr/bin/env bash
# Environment setup for fellowship project

# Add project paths to PATH
export PATH="metatoolkit/:$PATH"
export PATH="maaslin3/R/:$PATH"

# Activate Python virtual environment
if [ -f venv/bin/activate ]; then
    . venv/bin/activate
else
    echo "⚠️  No venv found at venv/bin/activate"
fi

