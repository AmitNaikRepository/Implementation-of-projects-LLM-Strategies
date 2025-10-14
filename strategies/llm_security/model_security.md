# Model Security Best Practices

## Executive Summary

Model security encompasses protecting AI models throughout their entire lifecycle—from development and training to deployment and monitoring. This strategy provides comprehensive guidance on securing LLM infrastructure, preventing model theft, ensuring access control, and maintaining operational security in production environments.

**Business Impact**: Protects intellectual property, prevents unauthorized model access, ensures compliance with security standards, and maintains service reliability.

## Core Security Principles

### 1. Defense in Depth
Implement multiple layers of security controls to protect models at every stage.

### 2. Least Privilege Access
Grant minimum necessary permissions to users, services, and applications.

### 3. Zero Trust Architecture
Never trust, always verify—authenticate and authorize every request.

### 4. Continuous Monitoring
Real-time detection of anomalies, unauthorized access attempts, and performance degradation.

## Model Lifecycle Security

### Phase 1: Development and Training Security

#### Secure Training Environment

```python
# secure_training_config.py
import os
from dataclasses import dataclass
from typing import List, Dict
import hashlib
import json

@dataclass
class SecureTrainingConfig:
    """Configuration for secure model training"""

    model_name: str
    training_data_path: str
    output_path: str
    allowed_users: List[str]
    encryption_enabled: bool = True
    audit_logging: bool = True

    def validate_environment(self) -> Dict[str, bool]:
        """Validate training environment security"""
        checks = {
            "secure_storage": self._check_secure_storage(),
            "access_controls": self._check_access_controls(),
            "network_isolation": self._check_network_isolation(),
            "logging_enabled": self._check_logging(),
        }
        return checks

    def _check_secure_storage(self) -> bool:
        """Verify training data is in secure storage"""
        # Check if path is in secure, encrypted storage
        secure_prefixes = ['/secure/', '/encrypted/', 's3://secure-']
        return any(self.training_data_path.startswith(p) for p in secure_prefixes)

    def _check_access_controls(self) -> bool:
        """Verify proper access controls are in place"""
        # Check file permissions (Unix-like systems)
        if os.path.exists(self.training_data_path):
            stat_info = os.stat(self.training_data_path)
            # Check if file is not world-readable (mode & 0o004 == 0)
            return (stat_info.st_mode & 0o004) == 0
        return False

    def _check_network_isolation(self) -> bool:
        """Verify training environment is network-isolated"""
        # Check if running in isolated network
        # This is environment-specific
        isolated_networks = ['vpc-secure', 'training-isolated']
        network_id = os.getenv('NETWORK_ID', '')
        return network_id in isolated_networks

    def _check_logging(self) -> bool:
        """Verify audit logging is configured"""
        return self.audit_logging

    def generate_model_fingerprint(self, model_weights_path: str) -> str:
        """Generate cryptographic fingerprint of model"""
        sha256_hash = hashlib.sha256()
        with open(model_weights_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

# Usage
config = SecureTrainingConfig(
    model_name="customer-service-llm-v2",
    training_data_path="/secure/training/data",
    output_path="/secure/models/output",
    allowed_users=["data-scientist@company.com", "ml-engineer@company.com"]
)

# Validate before training
security_checks = config.validate_environment()
if not all(security_checks.values()):
    raise SecurityError(f"Security validation failed: {security_checks}")

# Generate fingerprint after training
model_fingerprint = config.generate_model_fingerprint("/secure/models/output/model.bin")
print(f"Model fingerprint: {model_fingerprint}")
```

#### Training Data Protection

```python
# data_protection.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import json
from typing import Dict, Any

class TrainingDataProtection:
    """Encrypt and protect sensitive training data"""

    def __init__(self, encryption_key: bytes = None):
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())

    def encrypt_dataset(self, data: Dict[str, Any]) -> bytes:
        """Encrypt training dataset"""
        json_data = json.dumps(data).encode()
        encrypted = self.cipher.encrypt(json_data)
        return encrypted

    def decrypt_dataset(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt training dataset"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted.decode())

    def sanitize_pii(self, text: str) -> str:
        """Remove or mask PII from training data"""
        import re

        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                     '[EMAIL]', text)

        # Phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                     '[PHONE]', text)

        # SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b',
                     '[SSN]', text)

        # Credit card numbers
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                     '[CREDIT_CARD]', text)

        return text

# Usage
protector = TrainingDataProtection()

training_data = {
    "examples": [
        {"text": "Customer John Doe email: john@example.com called about...",
         "label": "support_inquiry"}
    ]
}

# Sanitize PII
sanitized = protector.sanitize_pii(training_data["examples"][0]["text"])
training_data["examples"][0]["text"] = sanitized

# Encrypt before storage
encrypted = protector.encrypt_dataset(training_data)
```

### Phase 2: Model Storage and Versioning Security

```python
# model_registry.py
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import json

class SecureModelRegistry:
    """Secure model registry with versioning and access control"""

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.access_logs = []

    def register_model(
        self,
        model_id: str,
        version: str,
        model_path: str,
        metadata: Dict,
        authorized_users: List[str]
    ) -> Dict:
        """Register a new model version with security metadata"""

        # Generate model checksum
        checksum = self._compute_checksum(model_path)

        # Create model record
        model_record = {
            "model_id": model_id,
            "version": version,
            "path": model_path,
            "checksum": checksum,
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "authorized_users": authorized_users,
            "access_count": 0,
            "status": "active"
        }

        # Store securely
        self.storage.save(f"models/{model_id}/{version}", model_record)

        # Log registration
        self._log_access("REGISTER", model_id, version, "system")

        return model_record

    def get_model(
        self,
        model_id: str,
        version: str,
        requesting_user: str
    ) -> Optional[Dict]:
        """Retrieve model with access control and integrity verification"""

        # Load model record
        model_record = self.storage.load(f"models/{model_id}/{version}")

        if not model_record:
            self._log_access("ACCESS_DENIED", model_id, version,
                           requesting_user, "Model not found")
            return None

        # Verify authorization
        if requesting_user not in model_record["authorized_users"]:
            self._log_access("ACCESS_DENIED", model_id, version,
                           requesting_user, "Unauthorized")
            raise PermissionError(f"User {requesting_user} not authorized for model {model_id}")

        # Verify integrity
        current_checksum = self._compute_checksum(model_record["path"])
        if current_checksum != model_record["checksum"]:
            self._log_access("INTEGRITY_VIOLATION", model_id, version,
                           requesting_user, "Checksum mismatch")
            raise SecurityError("Model integrity check failed")

        # Update access tracking
        model_record["access_count"] += 1
        model_record["last_accessed"] = datetime.utcnow().isoformat()
        model_record["last_accessed_by"] = requesting_user

        # Log successful access
        self._log_access("ACCESS_GRANTED", model_id, version, requesting_user)

        return model_record

    def revoke_model(self, model_id: str, version: str, reason: str) -> None:
        """Revoke a model version (security incident response)"""
        model_record = self.storage.load(f"models/{model_id}/{version}")

        if model_record:
            model_record["status"] = "revoked"
            model_record["revoked_at"] = datetime.utcnow().isoformat()
            model_record["revocation_reason"] = reason
            self.storage.save(f"models/{model_id}/{version}", model_record)

            self._log_access("REVOKED", model_id, version, "system", reason)

    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA-256 checksum of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _log_access(
        self,
        action: str,
        model_id: str,
        version: str,
        user: str,
        details: str = ""
    ) -> None:
        """Log all model access attempts"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "model_id": model_id,
            "version": version,
            "user": user,
            "details": details
        }
        self.access_logs.append(log_entry)

        # In production: send to centralized logging
        print(f"[AUDIT] {json.dumps(log_entry)}")

# Usage
registry = SecureModelRegistry(storage_backend=SecureStorage())

# Register model
registry.register_model(
    model_id="gpt-custom-v1",
    version="1.0.0",
    model_path="/secure/models/gpt-custom-v1.bin",
    metadata={"architecture": "transformer", "params": "7B"},
    authorized_users=["api-service@company.com", "admin@company.com"]
)

# Retrieve model with access control
try:
    model = registry.get_model(
        model_id="gpt-custom-v1",
        version="1.0.0",
        requesting_user="api-service@company.com"
    )
except PermissionError as e:
    print(f"Access denied: {e}")
```

### Phase 3: Deployment Security

#### API Authentication and Authorization

```python
# api_security.py
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import jwt
import hashlib
from typing import Dict, Optional

app = FastAPI()
security = HTTPBearer()

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class ModelAPIAuth:
    """Authentication and authorization for model API"""

    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}  # In production: use database
        self.rate_limits: Dict[str, List[datetime]] = {}

    def create_access_token(
        self,
        user_id: str,
        scopes: List[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = {
            "sub": user_id,
            "scopes": scopes,
            "exp": datetime.utcnow() + (expires_delta or timedelta(minutes=30))
        }
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def check_rate_limit(
        self,
        user_id: str,
        max_requests: int = 100,
        window_minutes: int = 1
    ) -> bool:
        """Check if user has exceeded rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)

        # Get user's request history
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []

        # Clean old requests
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id]
            if req_time > window_start
        ]

        # Check limit
        if len(self.rate_limits[user_id]) >= max_requests:
            return False

        # Record new request
        self.rate_limits[user_id].append(now)
        return True

    def require_scopes(self, required_scopes: List[str]):
        """Dependency for scope-based authorization"""
        def verify_scopes(
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            token = credentials.credentials
            payload = self.verify_token(token)

            user_scopes = payload.get("scopes", [])
            for scope in required_scopes:
                if scope not in user_scopes:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing required scope: {scope}"
                    )

            # Check rate limit
            user_id = payload.get("sub")
            if not self.check_rate_limit(user_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )

            return payload

        return verify_scopes

auth = ModelAPIAuth()

# Protected endpoint example
@app.post("/api/v1/generate")
async def generate_text(
    request: Dict,
    user = Depends(auth.require_scopes(["model:generate"]))
):
    """Protected model inference endpoint"""
    user_id = user.get("sub")

    # Log request
    print(f"[API] User {user_id} requested generation")

    # Process request (pseudo-code)
    response = model.generate(request["prompt"])

    return {"response": response, "user_id": user_id}

# Admin endpoint example
@app.post("/api/v1/admin/revoke-model")
async def revoke_model(
    model_id: str,
    user = Depends(auth.require_scopes(["model:admin"]))
):
    """Admin endpoint to revoke compromised models"""
    registry.revoke_model(model_id, "1.0.0", "Security incident")
    return {"status": "revoked", "model_id": model_id}
```

#### Rate Limiting and Throttling

```typescript
// rate_limiter.ts
import { Request, Response, NextFunction } from 'express';

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  keyGenerator: (req: Request) => string;
}

class TokenBucket {
  private tokens: number;
  private lastRefill: number;

  constructor(
    private readonly capacity: number,
    private readonly refillRate: number // tokens per second
  ) {
    this.tokens = capacity;
    this.lastRefill = Date.now();
  }

  tryConsume(count: number = 1): boolean {
    this.refill();

    if (this.tokens >= count) {
      this.tokens -= count;
      return true;
    }

    return false;
  }

  private refill(): void {
    const now = Date.now();
    const timePassed = (now - this.lastRefill) / 1000; // seconds
    const tokensToAdd = timePassed * this.refillRate;

    this.tokens = Math.min(this.capacity, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }
}

class ModelRateLimiter {
  private buckets: Map<string, TokenBucket> = new Map();

  constructor(
    private readonly requestsPerSecond: number = 10,
    private readonly burstCapacity: number = 20
  ) {}

  middleware() {
    return (req: Request, res: Response, next: NextFunction) => {
      const key = this.getClientKey(req);

      // Get or create bucket for this client
      if (!this.buckets.has(key)) {
        this.buckets.set(
          key,
          new TokenBucket(this.burstCapacity, this.requestsPerSecond)
        );
      }

      const bucket = this.buckets.get(key)!;

      // Try to consume token
      if (bucket.tryConsume()) {
        next();
      } else {
        res.status(429).json({
          error: 'Rate limit exceeded',
          retryAfter: 1 / this.requestsPerSecond
        });
      }
    };
  }

  private getClientKey(req: Request): string {
    // Use API key or user ID if available, otherwise IP
    return req.headers['x-api-key'] as string ||
           req.ip ||
           'unknown';
  }
}

// Usage
import express from 'express';

const app = express();
const rateLimiter = new ModelRateLimiter(10, 20); // 10 req/s, burst of 20

app.use('/api/generate', rateLimiter.middleware());

app.post('/api/generate', (req, res) => {
  // Model inference logic
  res.json({ response: 'Generated text...' });
});
```

### Phase 4: Runtime Monitoring and Threat Detection

```python
# threat_detection.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict
import statistics

@dataclass
class SecurityEvent:
    timestamp: datetime
    event_type: str
    user_id: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    details: Dict

class ModelThreatDetector:
    """Real-time threat detection for model APIs"""

    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.user_behavior: Dict[str, List[Dict]] = defaultdict(list)
        self.baseline_metrics: Dict[str, float] = {}

    def detect_anomalies(
        self,
        user_id: str,
        request_data: Dict
    ) -> Optional[SecurityEvent]:
        """Detect anomalous behavior patterns"""

        # Track user behavior
        self.user_behavior[user_id].append({
            "timestamp": datetime.utcnow(),
            "request_size": len(str(request_data)),
            "endpoint": request_data.get("endpoint"),
        })

        # Detect unusual request patterns
        anomaly = self._check_behavioral_anomaly(user_id)
        if anomaly:
            return self._create_event(
                event_type="BEHAVIORAL_ANOMALY",
                user_id=user_id,
                severity="MEDIUM",
                details=anomaly
            )

        # Detect model extraction attempts
        extraction = self._detect_model_extraction(user_id, request_data)
        if extraction:
            return self._create_event(
                event_type="MODEL_EXTRACTION_ATTEMPT",
                user_id=user_id,
                severity="HIGH",
                details=extraction
            )

        return None

    def _check_behavioral_anomaly(self, user_id: str) -> Optional[Dict]:
        """Detect unusual user behavior"""
        recent_requests = self.user_behavior[user_id][-100:]

        if len(recent_requests) < 10:
            return None  # Not enough data

        # Check request frequency
        recent_times = [r["timestamp"] for r in recent_requests[-10:]]
        time_diffs = [
            (recent_times[i+1] - recent_times[i]).total_seconds()
            for i in range(len(recent_times)-1)
        ]

        avg_interval = statistics.mean(time_diffs) if time_diffs else 0

        # Suspiciously fast requests (potential automated attack)
        if avg_interval < 0.1:  # Less than 100ms between requests
            return {
                "reason": "Unusually high request frequency",
                "avg_interval_seconds": avg_interval,
                "request_count": len(recent_requests)
            }

        # Check request size variation
        request_sizes = [r["request_size"] for r in recent_requests]
        if request_sizes:
            avg_size = statistics.mean(request_sizes)
            std_dev = statistics.stdev(request_sizes) if len(request_sizes) > 1 else 0

            # Large requests could indicate data exfiltration
            if avg_size > 10000 and std_dev > avg_size * 0.5:
                return {
                    "reason": "Unusual request size pattern",
                    "avg_size": avg_size,
                    "std_dev": std_dev
                }

        return None

    def _detect_model_extraction(
        self,
        user_id: str,
        request_data: Dict
    ) -> Optional[Dict]:
        """Detect potential model extraction attempts"""

        # Check for systematic probing patterns
        recent = self.user_behavior[user_id][-50:]

        if len(recent) < 20:
            return None

        # Look for patterns indicating extraction (e.g., sequential queries)
        prompts = [r.get("prompt", "") for r in recent if "prompt" in r]

        # Check for very similar sequential prompts (fuzzing)
        if len(prompts) >= 10:
            similar_count = sum(
                1 for i in range(len(prompts)-1)
                if self._similarity(prompts[i], prompts[i+1]) > 0.9
            )

            if similar_count > len(prompts) * 0.7:  # >70% similar
                return {
                    "reason": "Sequential similar prompts detected",
                    "similarity_ratio": similar_count / len(prompts)
                }

        return None

    def _similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simple version)"""
        if not str1 or not str2:
            return 0.0

        longer = max(len(str1), len(str2))
        if longer == 0:
            return 1.0

        # Simple character-based similarity
        matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        return matches / longer

    def _create_event(
        self,
        event_type: str,
        user_id: str,
        severity: str,
        details: Dict
    ) -> SecurityEvent:
        """Create and log security event"""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=user_id,
            severity=severity,
            details=details
        )

        self.events.append(event)

        # Alert on high/critical severity
        if severity in ["HIGH", "CRITICAL"]:
            self._send_alert(event)

        return event

    def _send_alert(self, event: SecurityEvent) -> None:
        """Send alert to security team"""
        print(f"[ALERT] {event.severity}: {event.event_type} - User: {event.user_id}")
        print(f"Details: {event.details}")

        # In production: integrate with PagerDuty, Slack, email, etc.

# Usage with API
detector = ModelThreatDetector()

@app.post("/api/generate")
async def generate_with_monitoring(request: Dict, user_id: str):
    # Check for threats
    threat = detector.detect_anomalies(user_id, request)

    if threat and threat.severity in ["HIGH", "CRITICAL"]:
        raise HTTPException(status_code=403, detail="Suspicious activity detected")

    # Proceed with generation
    response = model.generate(request["prompt"])
    return {"response": response}
```

## Security Checklist

### Development Phase
- [ ] Use secure, isolated training environments
- [ ] Encrypt training data at rest and in transit
- [ ] Sanitize PII from training datasets
- [ ] Generate and store model fingerprints
- [ ] Implement version control for models
- [ ] Document model provenance and lineage

### Deployment Phase
- [ ] Implement strong authentication (OAuth2, JWT)
- [ ] Use API keys with proper rotation policies
- [ ] Enable rate limiting and throttling
- [ ] Set up comprehensive audit logging
- [ ] Implement scope-based authorization
- [ ] Use TLS 1.3 for all communications

### Operations Phase
- [ ] Monitor for anomalous access patterns
- [ ] Track model performance degradation
- [ ] Implement automated threat detection
- [ ] Maintain incident response procedures
- [ ] Regularly rotate secrets and keys
- [ ] Conduct security audits quarterly

### Compliance
- [ ] Document data handling procedures
- [ ] Implement data retention policies
- [ ] Maintain access audit trails
- [ ] Enable GDPR/CCPA compliance features
- [ ] Conduct regular penetration testing
- [ ] Maintain security certifications (SOC 2, ISO 27001)

## Incident Response Procedures

### Model Compromise Response

```python
# incident_response.py
from enum import Enum
from datetime import datetime
from typing import List, Dict

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ModelIncidentResponse:
    """Automated incident response for model security events"""

    def __init__(self, registry: SecureModelRegistry):
        self.registry = registry
        self.incidents: List[Dict] = []

    def handle_incident(
        self,
        incident_type: str,
        severity: IncidentSeverity,
        affected_models: List[str],
        details: Dict
    ) -> Dict:
        """Execute incident response procedures"""

        incident_id = self._create_incident(
            incident_type, severity, affected_models, details
        )

        # Execute response based on severity
        if severity == IncidentSeverity.CRITICAL:
            self._critical_response(affected_models, incident_id)
        elif severity == IncidentSeverity.HIGH:
            self._high_response(affected_models, incident_id)
        else:
            self._standard_response(affected_models, incident_id)

        return {"incident_id": incident_id, "status": "handled"}

    def _critical_response(self, models: List[str], incident_id: str):
        """Critical incident response: immediate lockdown"""

        for model_id in models:
            # Immediately revoke all model versions
            self.registry.revoke_model(
                model_id, "all",
                f"Critical security incident: {incident_id}"
            )

        # Alert security team
        self._alert_security_team("CRITICAL", incident_id, models)

        # Initiate forensic analysis
        self._start_forensics(incident_id)

    def _high_response(self, models: List[str], incident_id: str):
        """High severity response: isolate and investigate"""

        for model_id in models:
            # Temporarily disable affected models
            self.registry.revoke_model(
                model_id, "latest",
                f"High severity incident: {incident_id}"
            )

        # Alert ops team
        self._alert_ops_team("HIGH", incident_id, models)

    def _standard_response(self, models: List[str], incident_id: str):
        """Standard response: monitor and log"""
        # Increase monitoring
        # Log for investigation
        pass

    def _create_incident(
        self,
        incident_type: str,
        severity: IncidentSeverity,
        affected_models: List[str],
        details: Dict
    ) -> str:
        """Create incident record"""
        incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        incident = {
            "id": incident_id,
            "type": incident_type,
            "severity": severity.name,
            "affected_models": affected_models,
            "details": details,
            "created_at": datetime.utcnow().isoformat(),
            "status": "open"
        }

        self.incidents.append(incident)
        return incident_id

    def _alert_security_team(self, severity: str, incident_id: str, models: List[str]):
        """Send alert to security team"""
        print(f"[SECURITY ALERT] {severity} incident {incident_id}")
        print(f"Affected models: {models}")
        # In production: PagerDuty, Slack, email

    def _alert_ops_team(self, severity: str, incident_id: str, models: List[str]):
        """Send alert to ops team"""
        print(f"[OPS ALERT] {severity} incident {incident_id}")

    def _start_forensics(self, incident_id: str):
        """Initiate forensic analysis"""
        print(f"[FORENSICS] Starting analysis for {incident_id}")
        # Capture logs, snapshots, etc.
```

## Key Performance Indicators

### Security Metrics
- **Mean Time to Detect (MTTD)**: < 5 minutes
- **Mean Time to Respond (MTTR)**: < 15 minutes
- **False Positive Rate**: < 2%
- **Unauthorized Access Attempts**: 0 successful
- **Model Integrity Checks**: 100% passing

### Operational Metrics
- **API Availability**: 99.9% uptime
- **Authentication Latency**: < 10ms
- **Authorization Latency**: < 5ms
- **Security Overhead**: < 2% added latency

## Compliance and Auditing

```python
# audit_logger.py
from datetime import datetime
from typing import Dict, Any
import json

class SecurityAuditLogger:
    """Comprehensive audit logging for compliance"""

    def __init__(self, storage_backend):
        self.storage = storage_backend

    def log_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log access attempt with full context"""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "result": result,  # SUCCESS, DENIED, ERROR
            "ip_address": metadata.get("ip") if metadata else None,
            "user_agent": metadata.get("user_agent") if metadata else None,
            "session_id": metadata.get("session_id") if metadata else None,
            "metadata": metadata or {}
        }

        # Store in tamper-proof log
        self.storage.append_log(log_entry)

        # Send to SIEM
        self._send_to_siem(log_entry)

    def _send_to_siem(self, log_entry: Dict) -> None:
        """Send log entry to Security Information and Event Management system"""
        # Integration with Splunk, ELK, etc.
        print(f"[AUDIT] {json.dumps(log_entry)}")

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate compliance report for auditors"""

        logs = self.storage.query_logs(start_date, end_date)

        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_access_attempts": len(logs),
            "successful_accesses": len([l for l in logs if l["result"] == "SUCCESS"]),
            "denied_accesses": len([l for l in logs if l["result"] == "DENIED"]),
            "unique_users": len(set(l["user_id"] for l in logs)),
            "resources_accessed": list(set(l["resource"] for l in logs)),
        }

        return report
```

## Conclusion

Model security requires a comprehensive, multi-layered approach covering the entire AI lifecycle. By implementing these best practices—secure training environments, access controls, runtime monitoring, and incident response procedures—organizations can protect their AI assets while maintaining operational efficiency.

**Key Takeaways**:
1. Implement defense in depth with multiple security layers
2. Use strong authentication and authorization
3. Monitor continuously for threats and anomalies
4. Maintain comprehensive audit logs for compliance
5. Prepare incident response procedures in advance

---

*This document follows industry standards including OWASP AI Security guidelines, NIST AI Risk Management Framework, and SOC 2 requirements.*
