#!/bin/bash
# =============================================================================
# get_results.sh - Generate heatmaps, potential fields, and value maps
# =============================================================================
# Usage: ./get_results.sh --level_one --level_two --level_three ...
#
# This script processes specified levels and generates:
#   1. Phase heatmaps (generate_phase_heatmaps.py)
#   2. Potential field plots (plot_potential_fields.py)
#   3. Value map plots (plot_value_map.py)
#
# For both proposed_method and baseline_method directories.
# =============================================================================

# Don't use set -e as ((count++)) returns 1 when count is 0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Checkpoint step increment
CHECKPOINT_STEP=50

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_separator() {
    echo "============================================================================="
}

# Get available checkpoint numbers from models directory
# Returns space-separated list of checkpoint numbers divisible by CHECKPOINT_STEP
get_checkpoint_numbers() {
    local models_dir="$1"
    local prefix="$2"  # "adversary_sac" or "protagonist_sac"
    
    if [[ ! -d "$models_dir" ]]; then
        echo ""
        return
    fi
    
    # Extract checkpoint numbers from filenames, filter by step, sort numerically
    ls "$models_dir"/${prefix}_*.ckpt 2>/dev/null | \
        sed -n "s/.*${prefix}_\([0-9]*\)\.ckpt/\1/p" | \
        awk -v step="$CHECKPOINT_STEP" '$1 % step == 0' | \
        sort -n | \
        tr '\n' ' '
}

# Get common checkpoint numbers available for both adv and prt
get_common_checkpoints() {
    local models_dir="$1"
    
    local adv_ckpts=$(get_checkpoint_numbers "$models_dir" "adversary_sac")
    local prt_ckpts=$(get_checkpoint_numbers "$models_dir" "protagonist_sac")
    
    # Find intersection
    local common=""
    for ckpt in $adv_ckpts; do
        if echo "$prt_ckpts" | grep -qw "$ckpt"; then
            common="$common $ckpt"
        fi
    done
    
    echo $common
}

# Process a single method directory (proposed_method or baseline_method)
process_method() {
    local level="$1"
    local method="$2"
    local method_dir="$SCRIPT_DIR/$level/$method"
    local models_dir="$method_dir/${level}_models"
    
    if [[ ! -d "$method_dir" ]]; then
        log_warning "Directory not found: $method_dir"
        return
    fi
    
    print_separator
    log_info "Processing $level / $method"
    print_separator
    
    cd "$method_dir"
    
    # Step 0: Clean output directories
    log_step "Cleaning output directories..."
    local heatmap_dir="$method_dir/heatmap"
    local batches_dir="$heatmap_dir/batches"
    local value_map_dir="$heatmap_dir/value_map"
    local potential_fields_dir="$heatmap_dir/potential_fields"
    
    if [[ -d "$batches_dir" ]]; then
        rm -rf "$batches_dir"/*
        log_info "  Cleaned $batches_dir"
    fi
    if [[ -d "$value_map_dir" ]]; then
        rm -rf "$value_map_dir"/*
        log_info "  Cleaned $value_map_dir"
    fi
    if [[ -d "$potential_fields_dir" ]]; then
        rm -rf "$potential_fields_dir"/*
        log_info "  Cleaned $potential_fields_dir"
    fi
    log_success "Output directories cleaned"
    
    # Step 1: Generate phase heatmaps
    log_step "Generating phase heatmaps..."
    if [[ -f "generate_phase_heatmaps.py" ]]; then
        if python generate_phase_heatmaps.py 2>&1; then
            log_success "Phase heatmaps generated"
        else
            log_warning "Phase heatmaps generation failed or no data available"
        fi
    else
        log_warning "generate_phase_heatmaps.py not found"
    fi
    
    # Step 2: Plot potential fields
    log_step "Generating potential field plots..."
    if [[ -f "plot_potential_fields.py" ]] && [[ -d "$models_dir" ]]; then
        local common_ckpts=$(get_common_checkpoints "$models_dir")
        
        if [[ -z "$common_ckpts" ]]; then
            log_warning "No common checkpoints found for potential fields"
        else
            local count=0
            for ckpt in $common_ckpts; do
                log_info "  Plotting potential fields for checkpoint $ckpt..."
                if python plot_potential_fields.py --adv "$ckpt" --prt "$ckpt" 2>&1; then
                    count=$((count + 1))
                else
                    log_warning "  Failed for checkpoint $ckpt"
                fi
            done
            log_success "Generated $count potential field plots"
        fi
    else
        log_warning "plot_potential_fields.py not found or models directory missing"
    fi
    
    # Step 3: Plot value maps
    log_step "Generating value map plots..."
    if [[ -f "plot_value_map.py" ]] && [[ -d "$models_dir" ]]; then
        local prt_ckpts=$(get_checkpoint_numbers "$models_dir" "protagonist_sac")
        
        if [[ -z "$prt_ckpts" ]]; then
            log_warning "No protagonist checkpoints found for value maps"
        else
            local count=0
            for ckpt in $prt_ckpts; do
                log_info "  Plotting value map for checkpoint $ckpt..."
                if python plot_value_map.py --prt "$ckpt" 2>&1; then
                    count=$((count + 1))
                else
                    log_warning "  Failed for checkpoint $ckpt"
                fi
            done
            log_success "Generated $count value map plots"
        fi
    else
        log_warning "plot_value_map.py not found or models directory missing"
    fi
    
    cd "$SCRIPT_DIR"
    log_success "Completed $level / $method"
    echo ""
}

# Process a level (both proposed and baseline methods)
process_level() {
    local level="$1"
    
    echo ""
    print_separator
    echo -e "${GREEN}Processing Level: $level${NC}"
    print_separator
    echo ""
    
    # Process proposed method
    process_method "$level" "proposed_method"
    
    # Process baseline method
    process_method "$level" "baseline_method"
    
    log_success "Completed all processing for $level"
}

# =============================================================================
# Main Script
# =============================================================================

print_usage() {
    echo "Usage: $0 --level_one [--level_two] [--level_three] ..."
    echo ""
    echo "Available levels:"
    echo "  --level_one    Process level_one"
    echo "  --level_two    Process level_two"
    echo "  --level_three  Process level_three"
    echo "  --level_four   Process level_four"
    echo "  --level_five   Process level_five"
    echo "  --level_six    Process level_six"
    echo "  --all          Process all levels"
    echo ""
    echo "Example:"
    echo "  $0 --level_one --level_two"
    echo "  $0 --all"
}

# Parse arguments
LEVELS=()

if [[ $# -eq 0 ]]; then
    print_usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --level_one)
            LEVELS+=("level_one")
            shift
            ;;
        --level_two)
            LEVELS+=("level_two")
            shift
            ;;
        --level_three)
            LEVELS+=("level_three")
            shift
            ;;
        --level_four)
            LEVELS+=("level_four")
            shift
            ;;
        --level_five)
            LEVELS+=("level_five")
            shift
            ;;
        --level_six)
            LEVELS+=("level_six")
            shift
            ;;
        --all)
            LEVELS=("level_one" "level_two" "level_three" "level_four" "level_five" "level_six")
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ ${#LEVELS[@]} -eq 0 ]]; then
    log_error "No levels specified"
    print_usage
    exit 1
fi

# Print header
echo ""
print_separator
echo -e "${GREEN}   ADV-RL Results Generator${NC}"
print_separator
echo -e "Levels to process: ${CYAN}${LEVELS[*]}${NC}"
echo -e "Checkpoint step: ${CYAN}${CHECKPOINT_STEP}${NC}"
print_separator
echo ""

# Process each level
for level in "${LEVELS[@]}"; do
    process_level "$level"
done

# Print summary
echo ""
print_separator
echo -e "${GREEN}   All Processing Complete!${NC}"
print_separator
echo ""
