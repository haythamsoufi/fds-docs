# FDS RAG System Documentation

This directory contains comprehensive documentation for the FDS RAG (Retrieval-Augmented Generation) system upgrade.

## Documentation Overview

### 📋 [RAG Upgrade Plan](RAG_UPGRADE_PLAN.md)
Comprehensive overview of the RAG system upgrade including:
- System architecture changes
- Upgrade components and features
- Migration strategy
- Testing procedures
- Deployment plan
- Configuration reference
- Troubleshooting guide

### 🔄 [Migration Guide](MIGRATION_GUIDE.md)
Step-by-step instructions for migrating from legacy to enhanced RAG system:
- Pre-migration checklist
- Detailed migration steps
- Post-migration validation
- Performance testing
- Quality assurance procedures

### ⚠️ [Rollback Procedures](ROLLBACK_PROCEDURES.md)
Comprehensive rollback procedures for different scenarios:
- Emergency rollback (complete system failure)
- Gradual rollback (feature-specific issues)
- Data recovery procedures
- Automated rollback scripts
- Monitoring and alerting

## Quick Start

### For System Administrators
1. Read the [RAG Upgrade Plan](RAG_UPGRADE_PLAN.md) for system overview
2. Follow the [Migration Guide](MIGRATION_GUIDE.md) for step-by-step migration
3. Keep [Rollback Procedures](ROLLBACK_PROCEDURES.md) ready for emergencies

### For Developers
1. Review the architecture changes in [RAG Upgrade Plan](RAG_UPGRADE_PLAN.md)
2. Understand the migration process in [Migration Guide](MIGRATION_GUIDE.md)
3. Familiarize yourself with rollback procedures

### For DevOps Teams
1. Study the deployment plan in [RAG Upgrade Plan](RAG_UPGRADE_PLAN.md)
2. Prepare rollback scripts from [Rollback Procedures](ROLLBACK_PROCEDURES.md)
3. Set up monitoring and alerting systems

## System Components

### Core Services
- **Document Processing**: OCR, text extraction, chunking, deduplication
- **Embedding Service**: Dual embedding models (passage/query)
- **Vector Store**: ChromaDB with HNSW indexing
- **Retrieval Service**: Hybrid search (semantic + BM25)
- **Generation Service**: LLM integration with confidence calibration

### Key Features
- ✅ Advanced text splitting and normalization
- ✅ OCR for scanned PDFs and table extraction
- ✅ Dual embedding models with instruction formatting
- ✅ Hybrid retrieval with cross-encoder reranking
- ✅ Confidence calibration and no-answer thresholds
- ✅ Background processing with Celery
- ✅ Comprehensive admin endpoints
- ✅ Enhanced UI with citations and evidence blocks

## Migration Timeline

### Phase 1: Preparation (1-2 days)
- Environment setup
- Dependency installation
- Backup creation
- Configuration preparation

### Phase 2: Migration (2-3 days)
- Database schema migration
- Service deployment
- Document reprocessing
- Vector store migration

### Phase 3: Validation (1-2 days)
- Functional testing
- Performance validation
- Quality assurance
- User acceptance testing

### Phase 4: Go-Live (1 day)
- Production deployment
- Monitoring setup
- Performance optimization
- Documentation updates

## Emergency Contacts

- **Technical Lead**: [Contact Information]
- **DevOps Team**: [Contact Information]
- **Emergency Hotline**: [Contact Information]

## Support Resources

### Internal Resources
- Technical documentation: This directory
- Code repository: [Repository URL]
- Monitoring dashboards: [Dashboard URL]
- Issue tracking: [Issue Tracker URL]

### External Resources
- ChromaDB Documentation: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- React Documentation: https://reactjs.org/docs/

## Version Information

- **Current Version**: v2.0.0 (Enhanced RAG)
- **Previous Version**: v1.2.0 (Legacy RAG)
- **Documentation Version**: 1.0.0
- **Last Updated**: [Current Date]

## Contributing

To update this documentation:
1. Make changes to the relevant markdown files
2. Test all procedures and commands
3. Update version information
4. Submit pull request for review
5. Update this README if adding new documents

## License

This documentation is part of the FDS RAG System and follows the same license terms as the main project.
