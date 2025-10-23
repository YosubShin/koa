#!/bin/bash
#
# Cleanup script for /mnt/lustre storage area
#
# USAGE:
#   ssh koa
#   bash ~/koa-ml/scripts/cleanup_storage.sh [OPTIONS]
#
# OPTIONS:
#   --dry-run       Show what would be deleted without actually deleting
#   --keep-latest N Keep the N most recent jobs (default: 3)
#   -h, --help      Show help message

set -euo pipefail

REMOTE_DATA_ROOT="${KOA_ML_DATA_ROOT:-/mnt/lustre/koa/scratch/$USER/koa-ml}"
DRY_RUN=false
KEEP_LATEST=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --keep-latest) KEEP_LATEST="$2"; shift 2 ;;
        -h|--help)
            echo "KOA-ML Storage Cleanup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run           Show what would be deleted without deleting"
            echo "  --keep-latest N     Keep N most recent jobs (default: 3)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                    # See what would be deleted"
            echo "  $0 --keep-latest 5              # Keep 5 most recent jobs"
            echo "  $0 --dry-run --keep-latest 2    # Preview keeping only 2 jobs"
            exit 0
            ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "KOA-ML Storage Cleanup"
echo "======================================================================"
echo "Storage location: $REMOTE_DATA_ROOT"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE: No files will be deleted"
    echo ""
fi

# Function to cleanup a directory
cleanup_dir() {
    local base_dir="$1"
    local category="$2"

    if [ ! -d "$base_dir" ]; then
        echo "No $category directory found"
        return
    fi

    echo ""
    echo "Cleaning $category"
    echo "----------------------------------------------------------------------"

    # Count total directories
    local total=$(find "$base_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

    echo "Total jobs: $total"
    echo "Keeping: $KEEP_LATEST most recent"
    echo ""

    if [ "$total" -eq 0 ]; then
        echo "No jobs found"
        return
    fi

    local count=0
    local kept=0
    local deleted=0

    # Get sorted list of directories to a temp file to avoid pipe issues
    local tmpfile=$(mktemp)
    cd "$base_dir" && ls -t > "$tmpfile"

    # Process each directory
    while read -r dir; do
        local full_path="$base_dir/$dir"

        # Skip if not a directory
        if [ ! -d "$full_path" ]; then
            continue
        fi

        count=$((count + 1))
        local size=$(du -sh "$full_path" 2>/dev/null | awk '{print $1}')

        if [ $count -le $KEEP_LATEST ]; then
            echo "  ✓ KEEP   $dir ($size)"
            kept=$((kept + 1))
        else
            echo "  ✗ DELETE $dir ($size)"
            if [ "$DRY_RUN" = false ]; then
                rm -rf "$full_path"
            fi
            deleted=$((deleted + 1))
        fi
    done < "$tmpfile"

    rm -f "$tmpfile"

    echo ""
    if [ $total -gt 0 ]; then
        echo "Summary: Kept $kept, Deleted $deleted"
    else
        echo "No jobs found"
    fi
}

# Show current usage
echo "Current Storage Usage:"
echo "----------------------------------------------------------------------"
if [ -d "$REMOTE_DATA_ROOT/train/results" ]; then
    TRAIN_SIZE=$(du -sh "$REMOTE_DATA_ROOT/train/results" 2>/dev/null | awk '{print $1}')
    TRAIN_COUNT=$(ls "$REMOTE_DATA_ROOT/train/results" 2>/dev/null | wc -l)
    echo "Training results:   $TRAIN_SIZE ($TRAIN_COUNT jobs)"
fi

if [ -d "$REMOTE_DATA_ROOT/eval/results" ]; then
    EVAL_SIZE=$(du -sh "$REMOTE_DATA_ROOT/eval/results" 2>/dev/null | awk '{print $1}')
    EVAL_COUNT=$(ls "$REMOTE_DATA_ROOT/eval/results" 2>/dev/null | wc -l)
    echo "Evaluation results: $EVAL_SIZE ($EVAL_COUNT jobs)"
fi

TOTAL_SIZE=$(du -sh "$REMOTE_DATA_ROOT" 2>/dev/null | awk '{print $1}')
echo "Total:              $TOTAL_SIZE"

# Cleanup
cleanup_dir "$REMOTE_DATA_ROOT/train/results" "Training Results"
cleanup_dir "$REMOTE_DATA_ROOT/eval/results" "Evaluation Results"

# Final status
echo ""
echo "======================================================================"
echo "Cleanup Complete!"
echo "======================================================================"

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Final Storage Usage:"
    echo "----------------------------------------------------------------------"
    if [ -d "$REMOTE_DATA_ROOT/train/results" ]; then
        TRAIN_SIZE=$(du -sh "$REMOTE_DATA_ROOT/train/results" 2>/dev/null | awk '{print $1}')
        TRAIN_COUNT=$(ls "$REMOTE_DATA_ROOT/train/results" 2>/dev/null | wc -l)
        echo "Training results:   $TRAIN_SIZE ($TRAIN_COUNT jobs)"
    fi

    if [ -d "$REMOTE_DATA_ROOT/eval/results" ]; then
        EVAL_SIZE=$(du -sh "$REMOTE_DATA_ROOT/eval/results" 2>/dev/null | awk '{print $1}')
        EVAL_COUNT=$(ls "$REMOTE_DATA_ROOT/eval/results" 2>/dev/null | wc -l)
        echo "Evaluation results: $EVAL_SIZE ($EVAL_COUNT jobs)"
    fi

    TOTAL_SIZE=$(du -sh "$REMOTE_DATA_ROOT" 2>/dev/null | awk '{print $1}')
    echo "Total:              $TOTAL_SIZE"
else
    echo ""
    echo "This was a dry run. Use without --dry-run to actually delete files."
fi

echo ""
