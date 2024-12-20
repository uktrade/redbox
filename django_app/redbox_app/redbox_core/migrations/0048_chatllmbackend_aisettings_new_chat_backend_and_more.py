# Generated by Django 5.1.1 on 2024-10-08 14:23

import django.db.models.deletion
from django.db import migrations, models


def back_populate_chat_llm_backend(apps, schema_editor):
    ChatLLMBackend = apps.get_model("redbox_core", "ChatLLMBackend")

    is_default = True
    for model in "AISettings", "Chat":
        Model = apps.get_model("redbox_core", model)
        for ai_settings in Model.objects.all():
            try:
                ai_settings.new_chat_backend = ChatLLMBackend.objects.get(name=ai_settings.chat_backend)
            except ChatLLMBackend.DoesNotExist:
                if ai_settings.chat_backend.startswith("gpt-"):
                    provider = "azure_openai"
                elif ai_settings.chat_backend.startswith("anthropic."):
                    provider = "bedrock"
                else:
                    provider = "openai"
                ai_settings.new_chat_backend = ChatLLMBackend.objects.create(name=ai_settings.chat_backend, provider=provider, is_default=is_default)
                is_default = False
            ai_settings.save()


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0047_aisettings_agentic_give_up_question_prompt_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatLLMBackend',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='The name of the model, e.g. “gpt-4o”, “claude-3-opus-20240229”.', max_length=128)),
                ('provider', models.CharField(choices=[('openai', 'Openai'), ('anthropic', 'Anthropic'), ('azure_openai', 'Azure Openai'), ('google_vertexai', 'Google Vertexai'), ('google_genai', 'Google Genai'), ('bedrock', 'Bedrock'), ('bedrock_converse', 'Bedrock Converse'), ('cohere', 'Cohere'), ('fireworks', 'Fireworks'), ('together', 'Together'), ('mistralai', 'Mistralai'), ('huggingface', 'Huggingface'), ('groq', 'Groq'), ('ollama', 'Ollama')], help_text='The model provider', max_length=128)),
                ('description', models.TextField(blank=True, help_text='brief description of the model', null=True)),
                ('is_default', models.BooleanField(default=False, help_text='is this the default llm to use.')),
                ('enabled', models.BooleanField(default=True, help_text='is this model enabled.')),
            ],
            options={
                'constraints': [models.UniqueConstraint(fields=('name', 'provider'), name='unique_name_provider')],
            },
        ),
        migrations.AddField(
            model_name='aisettings',
            name='new_chat_backend',
            field=models.ForeignKey(blank=True, help_text='LLM to use in chat', null=True, on_delete=django.db.models.deletion.CASCADE, to='redbox_core.chatllmbackend'),
        ),
        migrations.AddField(
            model_name='chat',
            name='new_chat_backend',
            field=models.ForeignKey(blank=True, help_text='LLM to use in chat', null=True, on_delete=django.db.models.deletion.CASCADE, to='redbox_core.chatllmbackend'),
        ),

        migrations.RunPython(back_populate_chat_llm_backend, migrations.RunPython.noop),

        migrations.RemoveField(
            model_name='aisettings',
            name='chat_backend',
        ),
        migrations.RemoveField(
            model_name='chat',
            name='chat_backend',
        ),
        migrations.RenameField(
            model_name='aisettings',
            old_name='new_chat_backend',
            new_name='chat_backend',
        ),
        migrations.RenameField(
            model_name='chat',
            old_name='new_chat_backend',
            new_name='chat_backend',
        ),
        migrations.AlterField(
            model_name='aisettings',
            name='chat_backend',
            field=models.ForeignKey(help_text='LLM to use in chat', on_delete=django.db.models.deletion.CASCADE,
                                    to='redbox_core.chatllmbackend'),
        ),
        migrations.AlterField(
            model_name='chat',
            name='chat_backend',
            field=models.ForeignKey(help_text='LLM to use in chat', on_delete=django.db.models.deletion.CASCADE,
                                    to='redbox_core.chatllmbackend'),
        ),
    ]
