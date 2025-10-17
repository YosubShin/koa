# Testing Guide for koa-ml

This guide walks you through testing your koa-ml installation and development workflow.

## Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all unit tests (no KOA connection needed)
pytest tests/ -m "not integration"

# Run unit tests with coverage
pytest tests/ -m "not integration" --cov=koa_ml --cov-report=term-missing
```

## Test Organization

The test suite is organized into three categories:

### 1. Unit Tests (No KOA Connection Required)

These tests use mocks and don't require actual KOA credentials or connectivity:

- **[tests/test_config.py](tests/test_config.py)** - Configuration loading and validation
- **[tests/test_ssh.py](tests/test_ssh.py)** - SSH command construction and execution
- **[tests/test_slurm.py](tests/test_slurm.py)** - Slurm job management logic

Run unit tests:
```bash
pytest tests/ -m "not integration" -v
```

### 2. Integration Tests (Require KOA Connection)

These tests connect to the actual KOA cluster and are marked with `@pytest.mark.integration`:

- **[tests/test_integration.py](tests/test_integration.py)** - End-to-end testing with real KOA

Run integration tests (requires `~/.config/koa-ml/config.yaml`):
```bash
pytest tests/test_integration.py -v
```

Skip slow tests:
```bash
pytest tests/test_integration.py -v -m "not slow"
```

### 3. Manual Verification Tests

Tests marked with `@pytest.mark.manual` submit real jobs and require manual verification:

```bash
pytest tests/test_integration.py -v -m manual
```

## Running Tests

### Run All Unit Tests

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run unit tests only
pytest tests/ -m "not integration"
```

### Run All Tests (Including Integration)

```bash
# Make sure you have KOA credentials configured
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test only configuration
pytest tests/test_config.py -v

# Test only SSH functionality
pytest tests/test_ssh.py -v

# Test only Slurm operations
pytest tests/test_slurm.py -v
```

### Run Specific Tests

```bash
# Run a specific test class
pytest tests/test_config.py::TestLoadConfig -v

# Run a specific test function
pytest tests/test_config.py::TestLoadConfig::test_load_basic_config -v
```

## Manual Testing Workflow

After running unit tests, verify the actual functionality with KOA:

### 1. Verify Configuration

```bash
# Check that your config file is valid
koa-ml check
```

Expected output:
```
== hostname ==
koa001  (or similar)
== sinfo ==
PARTITION AVAIL  TIMELIMIT  NODES GRES...
```

### 2. Test Job Submission

```bash
# Submit the example test job
koa-ml submit eval/scripts/example_job.slurm --partition gpu-sandbox --gpus 1
```

Expected output:
```
Submitted KOA job 12345
```

### 3. Check Job Status

```bash
# List your active jobs
koa-ml jobs
```

Expected output:
```
JOBID|NAME|STATE|TIME|TIME_LIMIT|NODES|NODELIST
12345|koa-test|RUNNING|00:01:23|00:10:00|1|gpu001
```

### 4. Cancel the Job

```bash
# Cancel the test job
koa-ml cancel 12345
```

Expected output:
```
Cancelled KOA job 12345
```

### 5. Verify Cancellation

```bash
# Check that the job is gone or cancelled
koa-ml jobs
```

## Integration Test Details

### Prerequisites for Integration Tests

1. **KOA Credentials**: You must have valid KOA credentials and SSH access
2. **Configuration File**: Create `~/.config/koa-ml/config.yaml` from the example
3. **SSH Access**: Ensure you can `ssh user@koa.its.hawaii.edu` successfully
4. **Network Access**: Must be on campus network or VPN if required

### What Integration Tests Do

- **Health Checks**: Verify SSH connectivity and basic Slurm commands
- **Job Lifecycle**: Submit a real job, verify it appears in queue, cancel it
- **Resource Requests**: Test GPU and CPU resource allocation
- **Custom Options**: Test custom remote filenames and sbatch arguments

### Integration Test Safety

- All integration tests clean up after themselves by cancelling submitted jobs
- Jobs use the `sandbox` partition with short time limits (2 minutes)
- If a test fails, jobs are still cancelled in the `finally` block

## Debugging Failed Tests

### Unit Test Failures

```bash
# Run with more verbose output
pytest tests/test_config.py -vv

# Show print statements
pytest tests/test_config.py -v -s

# Stop at first failure
pytest tests/test_config.py -v -x

# Drop into debugger on failure
pytest tests/test_config.py -v --pdb
```

### Integration Test Failures

If integration tests fail:

1. **Check KOA Connectivity**:
   ```bash
   ssh user@koa.its.hawaii.edu hostname
   ```

2. **Verify Configuration**:
   ```bash
   cat ~/.config/koa-ml/config.yaml
   ```

3. **Check Slurm Status**:
   ```bash
   ssh user@koa.its.hawaii.edu sinfo
   ```

4. **Check for Stuck Jobs**:
   ```bash
   koa-ml jobs
   # Cancel any test jobs manually
   koa-ml cancel <job_id>
   ```

## Continuous Testing During Development

### Watch Mode

Use `pytest-watch` for continuous testing during development:

```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and re-run tests
ptw tests/ -m "not integration"
```

### Pre-commit Hook

Create `.git/hooks/pre-commit` to run tests before committing:

```bash
#!/bin/bash
source .venv/bin/activate
pytest tests/ -m "not integration" --tb=short
exit $?
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Coverage Reports

Check test coverage:

```bash
# Install pytest-cov first
pip install pytest-cov

# Generate coverage report
pytest tests/ -m "not integration" --cov=koa_ml --cov-report=html

# Open the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Performance

Time your tests:

```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Time individual tests
pytest tests/test_config.py -v --durations=0
```

## Common Issues

### Issue: "Config file not found"

**Solution**: Create the config file:
```bash
mkdir -p ~/.config/koa-ml
cp .koa-config.example.yaml ~/.config/koa-ml/config.yaml
# Edit with your credentials
```

### Issue: Integration tests hang

**Solution**: Check SSH connectivity:
```bash
ssh -v user@koa.its.hawaii.edu
```

### Issue: "Permission denied" errors

**Solution**: Check your SSH key:
```bash
ssh-add -l  # List loaded keys
ssh-add ~/.ssh/id_rsa  # Add your key if needed
```

### Issue: Tests pass but CLI doesn't work

**Solution**: Reinstall in editable mode:
```bash
pip install -e .
```

## Next Steps

After all tests pass:

1. Submit a real training job on KOA
2. Monitor job status and output
3. Experiment with different partitions and resource requests
4. Add custom job templates for your workflows

## Getting Help

- For test failures: Check the error message and consult this guide
- For KOA issues: Contact uh-hpc-help@lists.hawaii.edu
- For koa-ml bugs: File an issue on the repository
