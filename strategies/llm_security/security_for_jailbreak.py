import hashlib
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

# Data structures
@dataclass
class ViolationRecord:#its the violation rector for each user 
    user_id: str
    timestamp: datetime
    violation_type: str  # jailbreak, harmful_content, spam, etc.
    severity: int  # 1-10 scale
    input_text_hash: str  # hashed for privacy
    metadata: Dict
#this will tell you the status of an account     
class AccountStatus(Enum):
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    TEMPORARILY_SUSPENDED = "temporarily_suspended"
    PERMANENTLY_BANNED = "permanently_banned"
    FLAGGED_FOR_MANUAL_REVIEW = "flagged_for_manual_review"
#if lets say a user is violating the terms of service what action should be taken
class ActionType(Enum):
    WARNING = "warning"
    TEMPORARY_RESTRICTION = "temporary_restriction"
    ACCOUNT_SUSPENSION = "account_suspension"
    EXTENDED_SUSPENSION = "extended_suspension"
    PERMANENT_BAN_REVIEW = "permanent_ban_review"

#if we want to take action what would this data structure would do 
@dataclass
class AccountAction:
    action_type: ActionType
    duration_hours: Optional[int] = None
    requires_human_review: bool = False
    requires_manager_approval: bool = False
    message: str = ""

class ViolationTracker:
    def __init__(self, database, security_queue):
        self.database = database
        self.security_queue = security_queue
        self.security_salt = "your_security_salt_here"
    
    def track_violation_attempt(self, user_id: str, violation_data: Dict) -> AccountAction:
        """Main function to track violation and determine action"""
        
        # Create violation record
        violation = ViolationRecord(
            user_id=user_id,
            timestamp=datetime.now(),
            violation_type=violation_data['violation_type'],
            severity=violation_data['severity'],
            input_text_hash=self.hash_sensitive_content(violation_data['input_text']),
            metadata=violation_data.get('metadata', {})
        )
        
        # Store in database
        self.database.insert_violation(violation)
        
        # Get user's violation history
        user_history = self.get_user_violation_history(user_id)
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(user_history, violation)
        
        # Determine action
        action = self.determine_account_action(risk_score, user_history)
        
        # Execute action
        self.execute_account_action(user_id, action)
        
        # Alert security team if needed
        if action.requires_human_review:
            self.alert_security_team(user_id, violation, action)
        
        return action
    
    def calculate_risk_score(self, user_history: List[ViolationRecord], 
                           current_violation: ViolationRecord) -> float:
        """Calculate user risk score based on history and current violation"""
        
        base_score = current_violation.severity
        
        # Frequency multiplier - recent violations in last 24 hours
        recent_violations = self.count_violations_in_time_window(
            user_history, hours=24
        )
        frequency_multiplier = min(recent_violations * 0.5, 3.0)
        
        # Pattern detection multiplier
        pattern_multiplier = self.detect_suspicious_patterns(user_history)
        
        # Repeat offender multiplier
        total_violations = len(user_history)
        repeat_offender_multiplier = 2.0 if total_violations > 5 else 1.0
        
        final_score = (base_score * frequency_multiplier * 
                      pattern_multiplier * repeat_offender_multiplier)
        
        return min(final_score, 10.0)
    
    def determine_account_action(self, risk_score: float, 
                               user_history: List[ViolationRecord]) -> AccountAction:
        """Determine what action to take based on risk score"""
        
        if risk_score <= 3:
            return AccountAction(
                action_type=ActionType.WARNING,
                message="Please follow our community guidelines",
                requires_human_review=False
            )
        
        elif risk_score <= 5:
            return AccountAction(
                action_type=ActionType.TEMPORARY_RESTRICTION,
                duration_hours=24,
                message="Account temporarily restricted for 24 hours",
                requires_human_review=False
            )
        
        elif risk_score <= 7:
            return AccountAction(
                action_type=ActionType.ACCOUNT_SUSPENSION,
                duration_hours=168,  # 7 days
                message="Account suspended for 7 days due to policy violations",
                requires_human_review=True
            )
        
        elif risk_score <= 9:
            return AccountAction(
                action_type=ActionType.EXTENDED_SUSPENSION,
                duration_hours=720,  # 30 days
                message="Account suspended for 30 days - manual review required",
                requires_human_review=True
            )
        
        else:  # risk_score > 9
            return AccountAction(
                action_type=ActionType.PERMANENT_BAN_REVIEW,
                message="Account flagged for permanent ban review",
                requires_human_review=True,
                requires_manager_approval=True
            )
    
    def detect_suspicious_patterns(self, user_history: List[ViolationRecord]) -> float:
        """Detect patterns that suggest coordinated or sophisticated attacks"""
        
        patterns = {
            'rapid_successive_attempts': 1.0,
            'escalating_complexity': 1.0,
            'multiple_violation_types': 1.0,
            'coordination_indicators': 1.0
        }
        
        # Check for rapid attempts (3+ in 10 minutes)
        rapid_attempts = self.count_violations_in_time_window(
            user_history, minutes=10
        )
        if rapid_attempts > 3:
            patterns['rapid_successive_attempts'] = 1.5
        
        # Check for escalating sophistication
        if self.is_escalating_sophistication(user_history):
            patterns['escalating_complexity'] = 1.3
        
        # Check for diverse attack vectors
        violation_types = set(v.violation_type for v in user_history)
        if len(violation_types) > 3:
            patterns['multiple_violation_types'] = 1.2
        
        # Look for coordination indicators
        if self.detect_coordinated_attack(user_history):
            patterns['coordination_indicators'] = 2.0
        
        return max(patterns.values())
    
    def execute_account_action(self, user_id: str, action: AccountAction):
        """Execute the determined account action"""
        
        if action.action_type == ActionType.WARNING:
            self.database.add_user_warning(user_id, action.message)
        
        elif action.action_type in [ActionType.TEMPORARY_RESTRICTION, 
                                   ActionType.ACCOUNT_SUSPENSION, 
                                   ActionType.EXTENDED_SUSPENSION]:
            suspension_until = datetime.now() + timedelta(hours=action.duration_hours)
            self.database.suspend_user(user_id, suspension_until, action.message)
        
        elif action.action_type == ActionType.PERMANENT_BAN_REVIEW:
            self.database.flag_user_for_ban_review(user_id)
        
        # Log the action
        self.database.log_account_action(user_id, action)
    
    def alert_security_team(self, user_id: str, violation: ViolationRecord, 
                          action: AccountAction):
        """Create alert for security team manual review"""
        
        priority = self.calculate_priority(violation.severity)
        
        review_ticket = {
            'user_id': user_id,
            'priority': priority,
            'violation_summary': {
                'type': violation.violation_type,
                'severity': violation.severity,
                'timestamp': violation.timestamp.isoformat()
            },
            'recommended_action': action.action_type.value,
            'user_history_summary': self.generate_user_summary(user_id),
            'created_at': datetime.now().isoformat(),
            'status': 'pending_review'
        }
        
        self.security_queue.add_ticket(review_ticket)
        
        # High priority violations get immediate alerts
        if priority == 'HIGH':
            self.send_immediate_alert(review_ticket)
    
    def hash_sensitive_content(self, content: str) -> str:
        """Hash content for security analysis while preserving privacy"""
        return hashlib.sha256((content + self.security_salt).encode()).hexdigest()
    
    def count_violations_in_time_window(self, user_history: List[ViolationRecord], 
                                      hours: int = None, minutes: int = None) -> int:
        """Count violations within specified time window"""
        
        if hours:
            time_threshold = datetime.now() - timedelta(hours=hours)
        elif minutes:
            time_threshold = datetime.now() - timedelta(minutes=minutes)
        else:
            return 0
        
        return sum(1 for v in user_history if v.timestamp > time_threshold)
    
    def get_user_violation_history(self, user_id: str) -> List[ViolationRecord]:
        """Get user's violation history from database"""
        return self.database.get_user_violations(user_id)
    
    def is_escalating_sophistication(self, user_history: List[ViolationRecord]) -> bool:
        """Check if violation attempts are becoming more sophisticated"""
        if len(user_history) < 3:
            return False
        
        # Sort by timestamp and check if recent violations have higher severity
        sorted_history = sorted(user_history, key=lambda x: x.timestamp)
        recent_avg = sum(v.severity for v in sorted_history[-3:]) / 3
        older_avg = sum(v.severity for v in sorted_history[:-3]) / len(sorted_history[:-3]) if len(sorted_history) > 3 else 0
        
        return recent_avg > older_avg + 1
    
    def detect_coordinated_attack(self, user_history: List[ViolationRecord]) -> bool:
        """Detect signs of coordinated attacks (simplified)"""
        # This would typically involve cross-user analysis
        # For now, just check for very rapid, similar violations
        if len(user_history) < 2:
            return False
        
        recent_violations = [v for v in user_history 
                           if v.timestamp > datetime.now() - timedelta(minutes=30)]
        
        return len(recent_violations) > 5
    
    def calculate_priority(self, severity: int) -> str:
        """Calculate priority level for security team"""
        if severity >= 8:
            return 'HIGH'
        elif severity >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_user_summary(self, user_id: str) -> Dict:
        """Generate summary of user for security team"""
        history = self.get_user_violation_history(user_id)
        
        return {
            'total_violations': len(history),
            'violation_types': list(set(v.violation_type for v in history)),
            'average_severity': sum(v.severity for v in history) / len(history) if history else 0,
            'first_violation': min(v.timestamp for v in history).isoformat() if history else None,
            'last_violation': max(v.timestamp for v in history).isoformat() if history else None
        }
    
    def send_immediate_alert(self, review_ticket: Dict):
        """Send immediate alert for high priority violations"""
        # Implementation would depend on your alerting system
        # Could be email, Slack, PagerDuty, etc.
        print(f"URGENT: High priority security violation - User {review_ticket['user_id']}")

# Main request handler integration
class SafeRequestHandler:
    def __init__(self, llm_model, llama_guard, violation_tracker):
        self.llm_model = llm_model
        self.llama_guard = llama_guard
        self.violation_tracker = violation_tracker
        self.database = violation_tracker.database
    
    def handle_user_request(self, user_input: str, user_id: str) -> str:
        """Main request handling with safety checks and violation tracking"""
        
        # Check account status first
        account_status = self.check_account_status_before_request(user_id)
        if account_status != "ALLOW_REQUEST":
            return self.get_blocked_message(account_status)
        
        try:
            # Input safety check
            safe_input = self.pre_process_request(user_input)
            
            # Generate response
            response = self.llm_model.generate(safe_input)
            
            # Output safety check
            safe_response = self.post_process_response(response, user_input)
            
            return safe_response
            
        except SafetyViolationException as violation:
            # Track the violation attempt
            self.violation_tracker.track_violation_attempt(user_id, {
                'violation_type': violation.category,
                'severity': violation.severity,
                'input_text': user_input,
                'metadata': self.get_current_request_metadata()
            })
            
            # Return appropriate message based on current account status
            account_status = self.database.get_account_status(user_id)
            if account_status == AccountStatus.TEMPORARILY_SUSPENDED:
                return "Your account is temporarily suspended due to policy violations."
            
            return "I can't help with that request."
    
    def pre_process_request(self, user_input: str) -> str:
        """Pre-process and validate user input"""
        # Basic sanitization
        cleaned_input = self.sanitize_input(user_input)
        
        # LlamaGuard safety check
        safety_result = self.llama_guard.classify_input(cleaned_input)
        if safety_result['is_unsafe']:
            raise SafetyViolationException(
                category=safety_result['violation_category'],
                severity=safety_result['severity']
            )
        
        return cleaned_input
    
    def post_process_response(self, llm_response: str, original_input: str) -> str:
        """Post-process and validate LLM response"""
        # LlamaGuard output check
        safety_result = self.llama_guard.classify_output(llm_response, original_input)
        if safety_result['is_unsafe']:
            raise SafetyViolationException(
                category=safety_result['violation_category'],
                severity=safety_result['severity']
            )
        
        return llm_response
    
    def check_account_status_before_request(self, user_id: str) -> str:
        """Check if user account allows requests"""
        status = self.database.get_account_status(user_id)
        
        if status == AccountStatus.TEMPORARILY_SUSPENDED:
            if self.is_suspension_expired(user_id):
                self.database.update_account_status(user_id, AccountStatus.ACTIVE)
                return "ALLOW_REQUEST"
            else:
                return "BLOCK_REQUEST_SUSPENDED"
        
        elif status == AccountStatus.UNDER_REVIEW:
            return "BLOCK_REQUEST_UNDER_REVIEW"
        
        elif status == AccountStatus.PERMANENTLY_BANNED:
            return "BLOCK_REQUEST_BANNED"
        
        else:
            return "ALLOW_REQUEST"
    
    def get_blocked_message(self, block_reason: str) -> str:
        """Get appropriate message for blocked requests"""
        messages = {
            "BLOCK_REQUEST_SUSPENDED": "Your account is temporarily suspended.",
            "BLOCK_REQUEST_UNDER_REVIEW": "Your account is under review.",
            "BLOCK_REQUEST_BANNED": "Your account has been permanently banned."
        }
        return messages.get(block_reason, "Access denied.")
    
    def sanitize_input(self, user_input: str) -> str:
        """Basic input sanitization"""
        # Remove or escape potentially dangerous characters
        return user_input.strip()
    
    def get_current_request_metadata(self) -> Dict:
        """Get current request metadata for logging"""
        return {
            'ip_address': 'user_ip_here',  # Would get from request
            'user_agent': 'user_agent_here',  # Would get from request headers
            'session_id': 'session_id_here',  # Would get from session
            'timestamp': datetime.now().isoformat()
        }
    
    def is_suspension_expired(self, user_id: str) -> bool:
        """Check if user's suspension has expired"""
        suspension_end = self.database.get_suspension_end_time(user_id)
        return datetime.now() > suspension_end if suspension_end else True

# Custom exception for safety violations
class SafetyViolationException(Exception):
    def __init__(self, category: str, severity: int):
        self.category = category
        self.severity = severity
        super().__init__(f"Safety violation: {category} (severity: {severity})")

# Usage example
if __name__ == "__main__":
    # Initialize components (these would be your actual implementations)
    database = None  # Your database implementation
    security_queue = None  # Your security team queue system
    llm_model = None  # Your custom LLM model
    llama_guard = None  # LlamaGuard implementation
    
    # Create violation tracker and request handler
    violation_tracker = ViolationTracker(database, security_queue)
    request_handler = SafeRequestHandler(llm_model, llama_guard, violation_tracker)
    
    # Handle a user request
    response = request_handler.handle_user_request(
        user_input="Some user input here",
        user_id="user123"
    )
    
    print(response)